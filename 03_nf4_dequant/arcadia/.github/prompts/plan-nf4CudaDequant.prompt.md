# Plan: NF4 CUDA 解量化 Kernel 实现

项目需要从零实现一个单核 NF4 解量化算子。采用自底向上的实现策略：先建立数据结构和 I/O 层，再实现核心 CUDA kernel，最后添加性能测量和验证。关键技术点包括：NF4 查找表硬编码、两级缩放公式、4-bit 解包、向量化内存写入（2x BF16 打包为 uint32_t）。

## Steps

### 1. 定义数据结构和常量

- 在 `include/` 创建 `nf4_types.h`：定义 `WeightMetadata` 结构体（num_rows, num_cols, blocksize, num_blocks, num_groups）
- 创建 `nf4_constants.cuh`：硬编码 16 个 NF4 查找表值（FP16 常数，来自标准正态分布的分位数）
- 创建 `common.h`：定义错误检查宏 `CHECK_CUDA`、数据类型别名

### 2. 实现 I/O 层

- `src/io/read_params.cpp`：解析文本格式的 `data/params.txt`，读取 blocksize、compute_type、target_gpu
- `src/io/read_weights.cpp`：读取二进制 `data/weights.bin` header（int64 num_rows, int64 num_cols, int32 blocksize）和数据块（packed_weights, absmax_q, absmax2, code2, offset）
- `src/io/write_output.cpp`：写入解量化后的 BF16/FP16 权重矩阵（行主序）和性能日志（JSON 或文本格式）

### 3. 实现 NF4 解量化 CUDA kernel

- 创建 `src/nf4_kernel.cu`
- 核函数签名：`__global__ void nf4_dequantize_kernel(const uint8_t* packed_weights, const uint8_t* absmax_q, const half* absmax2, const half* code2, float offset, half* output, int64_t num_rows, int64_t num_cols, int blocksize)`
- **线程组织**：1D 网格，每个线程处理 2 个元素（因为 4-bit 解包）
- **解量化步骤**：
  - 计算全局索引 → 确定块索引和组索引
  - 从 packed_weights 读取 1 字节 → 解包出 2 个 4-bit 索引（低 4 位、高 4 位）
  - 查 NF4 表 → 得到 2 个基础值
  - 计算两级缩放：`scale1 = code2[absmax_q[block_idx]]; scale2 = absmax2[group_idx]`
  - 公式：`value = nf4_lut[index] * scale1 * scale2 + offset`
  - 将 2 个 BF16 值打包为 `uint32_t`，一次性写入全局内存
- **边界处理**：在 kernel 末尾检查 `if (global_idx < total_elements)`

### 4. 实现主程序

- 创建 `src/main.cu`
- 流程：读取参数 → 读取权重 → 分配 GPU 内存 → 拷贝数据到 GPU → 配置 kernel（grid, block）→ 启动 kernel → 拷贝结果回 CPU → 写入输出 → 清理资源
- **Kernel 配置**：`threads_per_block = 256`，`num_blocks = (num_rows * num_cols / 2 + threads_per_block - 1) / threads_per_block`
- 使用 CUDA Events 测量 kernel 执行时间

### 5. 添加验证和性能测量

- 在 main 中加载 `data/reference.bin`（bitsandbytes 生成的参考结果）
- 计算 MAE（Mean Absolute Error）：遍历所有元素，累加 `|output[i] - reference[i]|`，除以总数
- 断言 `MAE < 1e-2`
- 计算有效带宽：`bandwidth_GB_s = (input_bytes + output_bytes) / (kernel_time_ms / 1000) / 1e9`
- 输出性能指标到文件和控制台

### 6. 配置 CMake 构建系统

- 更新 `CMakeLists.txt`：
  - `enable_language(CUDA)`，设置 CUDA 标准（C++17）
  - 添加 `include_directories(include/)`
  - 编译选项：`-O3 -use_fast_math -Xptxas=-v`（查看寄存器使用）
  - 添加可执行目标：`add_executable(nf4_dequant src/main.cu src/io/*.cpp src/nf4_kernel.cu)`
  - 链接 CUDA 库

### 7. 测试与调试

- 编译：`cd build && cmake .. && ninja`
- 运行：`./nf4_dequant`，检查输出文件和日志
- 验证 MAE 是否 < 1e-2
- 如果精度不足，检查：NF4 表值、缩放公式、4-bit 解包顺序（低位/高位）

### 8. 性能优化（可选进阶）

- 使用 shared memory 缓存 code2 和 absmax2 查找表
- 调整 block size（128/256/512）测试最优配置
- 考虑 warp-level 优化（__shfl 指令）
- 实验不同的内存访问模式（向量化加载）

## Verification

- 编译通过：`cd build && cmake .. && ninja`
- 运行：`./nf4_dequant`
- 检查输出：`output.bin` 文件大小应为 `num_rows * num_cols * 2` 字节（BF16）
- 验证精度：控制台输出 MAE < 1e-2
- 查看性能：日志文件显示 kernel 时间、带宽、加速比
- （可选）使用 `ncu` 分析：`ncu --set full ./nf4_dequant` 查看内存吞吐率和计算效率

## Decisions

- **内存访问策略**：选择打包写入（2x BF16 → uint32_t）而非独立写入，提升内存带宽利用率
- **线程映射**：每个线程处理 2 个输出元素（对应 1 个输入字节），简化索引计算并保证合并访问
- **NF4 表存储**：硬编码在常量内存或直接嵌入 kernel 代码，避免运行时加载开销
- **分块大小**：线程块 256（平衡占用率和寄存器压力），后续可调优

## Technical Details

### NF4 查找表值

NF4 是基于标准正态分布 N(0,1) 的分位数量化。16 个量化值对应将 CDF 等分为 16 个区间的边界值：

```
Index:  0     1     2     3     4     5     6     7     8     9     10    11    12    13    14    15
Value: -1.0 -0.6962 -0.5251 -0.3949 -0.2844 -0.1848 -0.0911 0.0   0.0796 0.1609 0.2461 0.3379 0.4409 0.5626 0.7229 1.0
```

### 二进制权重文件格式

```
[Header: 20 bytes]
  int64   num_rows      (8 bytes)
  int64   num_cols      (8 bytes)
  int32   blocksize     (4 bytes)

[Data: Variable size]
  uint8[] packed_weights    # size: num_rows * num_cols / 2
  uint8[] absmax_q          # size: num_blocks (= num_rows * num_cols / blocksize)
  fp16[]  absmax2           # size: num_groups (= num_blocks / 256)
  fp16[]  code2             # size: 256 (lookup table for absmax_q)
  float32 offset            # size: 1
```

### 解量化公式

对于每个元素：

1. 从 packed_weights 读取 4-bit 索引 `idx`（0-15）
2. 查表得到基础值：`base_val = NF4_LUT[idx]`
3. 计算块索引：`block_idx = element_idx / blocksize`
4. 计算组索引：`group_idx = block_idx / 256`
5. 一级缩放：`scale1 = code2[absmax_q[block_idx]]`
6. 二级缩放：`scale2 = absmax2[group_idx]`
7. 最终值：`output = base_val * scale1 * scale2 + offset`

### 4-bit 解包逻辑

每个 uint8 字节存储 2 个 4-bit 值：

```cpp
uint8_t packed = packed_weights[byte_idx];
uint8_t idx_low = packed & 0x0F;        // 低 4 位
uint8_t idx_high = (packed >> 4) & 0x0F; // 高 4 位
```

### 向量化内存写入

```cpp
half val1 = dequantize(idx_low, ...);
half val2 = dequantize(idx_high, ...);

// 打包为 uint32_t
uint32_t packed_output;
*reinterpret_cast<half*>(&packed_output) = val1;
*reinterpret_cast<half*>(reinterpret_cast<char*>(&packed_output) + 2) = val2;

// 一次性写入
*reinterpret_cast<uint32_t*>(&output[output_idx]) = packed_output;
```

## Implementation Notes

- CUDA 架构目标：至少 sm_70（Volta+）支持 FP16，sm_80（Ampere+）支持 BF16
- 如果 compute_type 是 "bf16"，使用 `__nv_bfloat16`；如果是 "fp16"，使用 `__half`
- 考虑使用 `__ldg()` 进行只读全局内存访问优化
- 边界检查应放在解量化计算之后，避免分支发散
- 可选：使用 `__launch_bounds__` 限制寄存器使用，提高占用率
