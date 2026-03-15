# NF4 Dequant CUDA 项目报告

## 1. 项目目标与背景

本项目实现了一个单核 CUDA NF4 解量化算子，将 4-bit NF4 量化权重直接恢复为 FP16/BF16 输出。NF4 的核心思想是利用权重近似服从正态分布这一先验，将标准正态分布的 CDF 等概率切分成 16 个区间，并以 16 个查找表常量完成 4-bit 索引到浮点值的映射。

相较于均匀量化，NF4 在分布中心区域分配更多量化等级，对大模型权重更友好。本项目的目标不是实现完整训练框架，而是聚焦一个高吞吐、单 kernel、可验证正确性的解量化内核，并给出对应的性能分析。

## 2. 实现思路

### 2.1 数据组织

量化权重由五部分组成：

1. `packed_weights`：每个字节存两个 4-bit NF4 索引。
2. `absmax_q`：每个 block 一个 `uint8` 级别的一级缩放索引。
3. `absmax2`：每 256 个 block 共用一个二级缩放因子，存为 FP16。
4. `code2`：长度为 256 的二级码表，用于把 `absmax_q` 还原成 block 级缩放值。
5. `offset`：缩放恢复时附加的常量偏移。

对应的数据结构定义在 `include/nf4_types.h` 中，解量化查找表 `NF4_LUT[16]` 定义在 `include/nf4_constants.cuh` 中。

### 2.2 解量化公式

单个 4-bit 索引的恢复可以表示为：

$$
scale_{block} = code2[absmax\_q[block\_idx]] \times absmax2[group\_idx] + offset
$$

$$
output = NF4\_LUT[idx] \times scale_{block}
$$

其中：

- `block_idx = elem_idx / blocksize`
- `group_idx = block_idx / 256`

这一路径对应 `src/nf4_kernel.cu` 中的 `process_packed_index`。该函数完成 4-bit 解包、索引查表、两级缩放恢复，以及最终输出写回。

### 2.3 主机侧执行流程

`src/main.cu` 的流程比较直接：

1. 读取 `params.txt` 与 `weights.bin`。
2. 分配 GPU 显存并拷贝 `packed_weights`、`absmax_q`、`absmax2`、`code2`。
3. 先做一次 warm-up kernel 调用。
4. 使用 CUDA Event 记录正式 kernel 时间。
5. 将结果拷回主机，与 `reference.bin` 计算 MAE。
6. 输出二进制结果和 `performance_log.json`。

## 3. 核心优化方法

### 3.1 4-bit 打包输入与成对输出写回

输入端每个字节包含两个 NF4 索引，天然把输入带宽压缩到原始 FP16 权重的四分之一。输出端没有逐元素写回，而是将两个 16-bit 输出打包为一个 `ushort2`，通过 `store_output_pair` 做一次 32-bit 对齐写回。这样做的目的有两个：

1. 降低写回指令数。
2. 提高 global store 的合并程度。

### 3.2 编译期块大小特化

kernel 没有在热点路径里做运行时整除，而是通过模板参数 `BLOCK_SHIFT` 对 blocksize 为 64、128、256 三种情况分别实例化。这样 `block_idx` 可以改写成右移位操作，减少整数除法开销。

### 3.3 两种调度内核

项目实现了两类 kernel：

1. `nf4_dequantize_kernel_strided`：使用 grid-stride 方式遍历 `packed_weights`，逻辑简单，适合小规模或不规则负载。
2. `nf4_dequantize_kernel_contiguous`：按 CTA tile 连续处理 packed bytes，并根据 `NF4_CONTIGUOUS_SMEM_LEVEL` 决定是否预取 `absmax2` 和 `absmax_q` 到 shared memory。

当前构建配置为：

- `NF4_USE_CODE2_CONST=ON`
- `NF4_SCHEDULE_MODE=force_contiguous`
- `NF4_CONTIGUOUS_SMEM_LEVEL=2`

这意味着实际跑的是 contiguous kernel，并且同时缓存 `absmax2` 和 `absmax_q` 元数据。

### 3.4 常量内存与 shared memory 协同

`code2` 可以通过 `cudaMemcpyToSymbolAsync` 拷入常量内存，减少热点路径中对全局内存码表的访问。另一方面，`SMEM_LEVEL=2` 会把当前 tile 涉及的一级、二级缩放元数据放入 shared memory，以降低重复访存成本。

这两者的 trade-off 很明确：

1. 常量内存能够减少一部分 LUT 访问压力。
2. shared memory 预取能改善局部性。
3. 但 shared memory 引入了额外同步与 scoreboard stall 风险。

### 3.5 FMA 与访存模式优化

根据项目已有优化记录，内核从早期版本演进到当前版本时，主要修正了两类问题：

1. warp 内线程对 packed bytes 的访问步长导致 global load/store 不够合并。
2. 非融合 FP32 指令过多，导致吞吐不足。

当前实现已经将 block scale 恢复写成 `fmaf(...)` 形式，降低了部分非融合浮点指令比例。

此外，当前实现修改了每个 warp 处理的范围为

$$tile\_base + i\cdot 256 + 0,\ 1,\ 2,\ \dots,\ 31$$

从而进行内存合并。

## 4. 实验环境与方法

### 4.1 硬件与软件环境

- GPU：NVIDIA GeForce RTX 4060 Laptop GPU
- 驱动：590.48.01
- 显存：8188 MiB
- Profiling 工具：`ncu`、`nsys`
- Python 工具：`uv 0.10.9`

需要说明的是，项目参数文件中的 `target_gpu="T4"` 和 CMake 中 `CMAKE_CUDA_ARCHITECTURES=75` 反映的是当前代码的目标配置，而不是本次实际测试机器的物理 GPU。实际测试是在 RTX 4060 Laptop GPU 上完成的。

### 4.2 数据与测量方法

本文使用了两组数据：

1. 仓库自带的 `4096 x 4096` 数据集。
2. 通过 `uv run python utils/datagen.py 1024 1024 64 --dtype fp16` 新生成的 `1024 x 1024` 独立数据集。

程序内部先进行一次 warm-up，再对正式一次 kernel 启动使用 CUDA Event 计时。因此表中的 kernel 时间是纯 kernel 时间，不包含文件读取、显存分配和主机侧校验。

有效带宽按下式计算：

$$
Bandwidth = \frac{InputBytes + OutputBytes}{KernelTime}
$$

其中输入包括 `packed_weights`、`absmax_q`、`absmax2`、`code2` 和 `offset`，输出为完整 FP16 结果矩阵。

## 5. 最终性能指标与分析

### 5.1 正确性结果

两组测试都通过了正确性验证，且误差远小于题目要求的 `1e-2`。

| 测试规模 | blocksize | 输出类型 | Kernel 时间 | 有效带宽 | MAE | 结论 |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 x 1024 | 64 | FP16 | 0.022528 ms | 117.12 GB/s | 2.365455e-05 | 通过 |
| 4096 x 4096 | 64 | FP16 | 0.278528 ms | 151.54 GB/s | 2.322402e-05 | 通过 |

### 5.2 性能分析

从上表可以看到三个清晰结论：

1. 解量化精度稳定。两组 MAE 都约为 `2.3e-05`，显著低于验收阈值，说明两级缩放恢复和 NF4 查表逻辑是正确的。
2. 大矩阵吞吐更高。`4096 x 4096` 的有效带宽达到 `151.54 GB/s`，明显高于 `1024 x 1024` 的 `117.12 GB/s`，说明更大的工作集能够更好摊薄 launch overhead，并让连续调度的访存模式发挥作用。
3. 内核本质上仍是带宽受限型工作负载。每个 packed byte 只对应有限浮点运算，但要读取多级元数据并写回 2 个 FP16 元素，因此性能更依赖访存效率而不是算力峰值。

同时，本次重新实测也证实了仓库中旧版 `performance_log.json` 的 `617.57 ms / 0.07 GB/s` 并不能代表当前实现状态。当前可复现的 4096² 结果已经回到 `0.28 ms` 量级，说明旧数据更可能来自历史构建、异常测量或未更新日志。

## 6. Nsight Systems 使用与分析

### 6.1 使用命令

本次使用的命令为：

```bash
nsys profile --stats=true -o /tmp/nf4_report_runs/nsys_4096_report ./nf4_dequant
```

### 6.2 关键结论

`nsys` 结果显示，单次程序运行的端到端耗时主要并不在 kernel 本身，而是在一次性资源准备和数据搬运上。

#### CUDA API 汇总

- `cudaMalloc`：65.07 ms，5 次调用，占 CUDA API 时间的 91.0%
- `cudaMemcpy`：3.59 ms，5 次调用，占 5.0%
- `cudaLaunchKernel`：0.285 ms，2 次调用，占 0.4%
- `cudaEventSynchronize`：0.30 ms，占 0.4%

#### GPU Memcpy 汇总

- H2D 总时间：713,486 ns，共 4 次，总传输 8.653 MB
- D2H 总时间：2,666,301 ns，共 1 次，总传输 33.554 MB

这说明在单次完整运行中，D2H 拷回完整输出矩阵的代价明显高于单次 kernel 计算本身。因此如果后续希望优化端到端性能，仅仅继续压缩 kernel 时间是不够的，还需要减少重复的显存分配和结果回传。

### 6.3 NSYS 结论

如果从“纯 kernel 基准”角度看，当前实现已经进入亚毫秒级；但如果从“完整程序一次运行”看，`cudaMalloc` 和 `cudaMemcpy` 仍然是不可忽视的固定成本。也就是说：

1. 当前实现的 kernel 优化已经有效。
2. 端到端优化空间主要在资源复用和流水化，而不是再去纠结一次启动中的几十微秒。

## 7. Nsight Compute 使用与分析

### 7.1 使用方式

尝试直接运行如下命令对当前环境重新采样：

```bash
sudo ncu --print-details all --set full --nvtx --call-stack -f -o nf4 ./nf4_dequant
```

### 7.2 从已有 `.ncu-rep` 导出的关键指标

从 `nf4.ncu-rep` 导出的结果可以确认当前热点 kernel 是：

```text
void nf4_dequantize_kernel_contiguous<__half, 6, 2>(...)
```

关键指标如下：

- Block Size：256
- Grid Size：8192
- Registers Per Thread：29
- Theoretical Occupancy：100%
- Achieved Occupancy：97.47%
- Achieved Active Warps Per SM：46.79
- Warp Cycles Per Issued Instruction：58.65
- Local Memory Spilling Requests：0
- Shared Memory Spilling Requests：0

这些数据说明当前 kernel 的 occupancy 已经很高，寄存器压力可控，而且没有发生局部内存或 shared memory spilling。换句话说，当前性能瓶颈已经不太像是“寄存器太多导致占用率不足”，而更像是访存和指令流水问题。

### 7.3 NCU 中反映出的计算与访存特征

导出报告还给出了一个很有价值的信号：

- Fused FP32 instructions：262,144
- Non-fused FP32 instructions：524,288

NCU 给出的建议是，如果进一步把非融合 FP32 指令压缩为融合形式，理论上仍有最多约 33% 的 FP32 吞吐提升空间。这和当前代码里已经把部分缩放恢复改写成 `fmaf` 的方向是一致的，也说明仍有残留的算术融合空间。

### 7.4 历史优化记录

以下是优化轨迹

1. 早期 contiguous 版本曾出现严重的 uncoalesced global access，L1TEX global load 只有 `6.8 / 32 bytes` 被有效利用，store 只有 `16 / 32 bytes`。
2. 通过[重排 warp 内处理顺序](#35-fma-与访存模式优化)后，global load 利用率先提升到 `11.6 / 32 bytes`，进一步提升到 `17.0 / 32 bytes`。
3. 打开更高等级 shared memory 预取后，短 scoreboard stall 依然存在，历史记录中该项平均达到 `17.9 cycles/warp`，约占每次发射间隔 `59.4 cycles` 的 30.1%。
4. 打开 `CODE2_CONST` 后又出现了轻微的 L2 slice workload imbalance，最大实例相对均值偏高约 `5.01%`。

这些现象共同说明：当前优化已经把问题从“完全不合并的访存”推进到了“高 occupancy 下的元数据访问和流水依赖”阶段。也就是说，下一步优化不应该再停留在宏观并行度，而要转向更细粒度的访存布局与流水调度。

## 8. 当前实现的优点与限制

### 8.1 优点

1. 正确性稳定，MAE 远低于题目要求。
2. 核函数时间很短，4096² 输入已经达到 `0.278528 ms`。
3. contiguous 调度加上 `SMEM_LEVEL=2` 的元数据缓存策略有效。
4. 高 occupancy、零 spilling，说明内核结构已经比较健康。

### 8.2 限制

1. 本次实际测试 GPU 是 RTX 4060 Laptop，但构建目标仍是 `sm_75`，没有针对当前硬件做原生架构编译。
2. 端到端运行中 `cudaMalloc` 和 D2H copy 的占比远高于单次 kernel，说明当前程序更像 microbenchmark，而不是完整推理流水。

## 9. 未来可继续提升的方向

1. **按真实 GPU 架构重新编译并做横向对比**。当前代码以 `sm_75` 为默认目标，后续应增加针对其他架构的原生构建，对比寄存器分配、指令调度和常量缓存行为。
2. **继续优化元数据访存模式**。当前主要压力不在主权重输入，而在 `absmax_q`、`absmax2` 与 `code2` 的多级索引恢复路径。可以继续尝试更紧凑的 tile 布局、更高效的 shared memory 映射或 warp 级广播。
3. **减少 shared memory 带来的 scoreboard stall**。历史 NCU 已经指出 `SMEM_LEVEL=1/2` 下存在 short scoreboard stall，后续可以进一步检查 bank conflict、同步粒度，以及是否能让编译器把部分热点值常驻寄存器。
4. **把端到端优化纳入目标**。`nsys` 已经表明一次性 `cudaMalloc` 和 D2H 拷回会放大总耗时，后续可以改成 buffer 复用、多次迭代平均计时，甚至在后续算子中直接消费解量化结果，避免把大矩阵拷回主机。
5. **补更丰富的测试矩阵**。当前本文给出了 1024² 和 4096² 两组数据，后续可以加入非对齐尺寸、不同 blocksize、FP16/BF16 双精度路径，以及 strided 与 contiguous 的 A/B 对比。

## 10. 结论

本项目已经完成了一个正确、快速、结构清晰的 NF4 CUDA 解量化实现。当前版本通过以下几类优化形成了较好的综合表现：

1. 4-bit packed 输入与成对输出写回。
2. 基于 `BLOCK_SHIFT` 的编译期特化。
3. contiguous tile 调度。
4. `code2` 常量内存路径。
5. `SMEM_LEVEL=2` 的元数据缓存。

重新测量后，`1024 x 1024` 的 kernel 时间为 `0.022528 ms`，`4096 x 4096` 的 kernel 时间为 `0.278528 ms`，对应带宽分别为 `117.12 GB/s` 和 `151.54 GB/s`，且 MAE 均维持在 `2.4e-05` 附近。NSYS 结果进一步说明，当前内核性能已经足够好，后续若要继续提升整体表现，应更多关注元数据访存、shared memory stall，以及端到端流水中的分配与拷贝成本。