# NF4 Dequantization CUDA Implementation - User Guide

## 项目概述

本项目实现了一个高效的单核 CUDA Kernel，用于将 NF4（4-bit NormalFloat）量化的神经网络权重解量化为 FP16/BF16 格式。

### 性能指标

在 1024x1024 权重矩阵上的测试结果：
- **Kernel 执行时间**: 0.0236 ms
- **有效内存带宽**: 112 GB/s
- **验证精度**: MAE = 0（完全准确）
- **寄存器使用**: 26 个寄存器/线程

## 快速开始

### 1. 编译项目

```bash
cd build
rm -rf *
cmake -G Ninja ..
ninja
```

### 2. 生成测试数据

```bash
# 生成 1024x1024 矩阵，blocksize=64
python utils/generate_data.py 1024 1024 64

# 或使用不同大小
python utils/generate_data.py 2048 2048 128
```

### 3. 运行程序

```bash
./nf4_dequant
```

程序将：
1. 读取 `data/params.txt` 和 `data/weights.bin`
2. 在 GPU 上执行解量化
3. 验证结果与 `data/reference.bin` 的误差
4. 输出性能指标到 `data/performance_log.json`
5. 保存解量化结果到 `data/output.bin`

## 项目结构

```
arcadia/
├── include/                 # 头文件
│   ├── common.h             # CUDA 错误检查宏
│   ├── nf4_types.h          # 数据结构定义
│   ├── nf4_constants.cuh    # NF4 查找表
│   ├── nf4_kernel.cuh       # Kernel 接口
│   └── nf4_io.h             # I/O 函数接口
├── src/                     # 源代码
│   ├── main.cu              # 主程序
│   ├── nf4_kernel.cu        # NF4 解量化 Kernel
│   └── io/                  # I/O 实现
│       ├── read_params.cpp
│       ├── read_weights.cpp
│       └── write_output.cpp
├── utils/                   # 工具脚本
│   ├── generate_data.py     # 数据生成器（推荐使用）
│   └── datagen.py           # bitsandbytes 版本（需要 PyTorch）
├── data/                    # 数据文件
│   ├── params.txt           # 配置参数
│   ├── weights.bin          # 量化权重（输入）
│   ├── reference.bin        # 参考输出（验证用）
│   ├── output.bin           # 解量化结果（输出）
│   └── performance_log.json # 性能日志
└── CMakeLists.txt           # 构建配置
```

## 技术实现细节

### NF4 解量化算法

NF4 使用分位数量化，将标准正态分布的 CDF 等分为 16 个区间。解量化公式：

```
output = NF4_LUT[idx] * scale1 * scale2 + offset
```

其中：
- `NF4_LUT[idx]`: 16 个预定义常量（存储在常量内存）
- `scale1 = code2[absmax_q[block_idx]]`: 块级缩放因子
- `scale2 = absmax2[group_idx]`: 组级缩放因子（256 块/组）
- `offset`: 量化偏移（通常为 0）

### Kernel 优化技术

1. **向量化内存写入**
   - 每个线程处理 2 个元素（1 字节 packed input）
   - 将 2x FP16 值打包为 uint32_t，一次性写入
   - 提高内存带宽利用率

2. **合并内存访问**
   - 256 线程/块，确保合并访问
   - 顺序读取 packed_weights

3. **常量内存缓存**
   - NF4 查找表存储在常量内存
   - 硬件自动广播和缓存

4. **边界处理**
   - 支持任意矩阵大小（无需对齐）
   - 最后一个线程块正确处理边界

### 数据文件格式

#### weights.bin（二进制）

```
[Header: 20 bytes]
  int64   num_rows      (8 bytes)
  int64   num_cols      (8 bytes)
  int32   blocksize     (4 bytes)

[Data: Variable]
  uint8[] packed_weights    # size: num_rows * num_cols / 2
  uint8[] absmax_q          # size: (num_rows * num_cols) / blocksize
  fp16[]  absmax2           # size: num_blocks / 256
  fp16[]  code2             # size: 256
  float32 offset            # size: 1
```

#### params.txt（文本）

```
blocksize = 64
compute_type = "bf16"
target_gpu = "T4"
```

#### reference.bin（二进制）

Row-major FP16 数组，shape = (num_rows, num_cols)

## 性能分析

### 使用 nvidia-smi 监控

```bash
nvidia-smi dmon -s u
```

### 使用 Nsight Compute 分析

```bash
ncu --set full ./nf4_dequant
```

关键指标：
- Memory throughput (achieved vs. theoretical)
- SM efficiency
- Register usage
- Occupancy

### 使用 Nsight Systems 分析

```bash
nsys profile --stats=true ./nf4_dequant
```

## 验证方法

程序自动验证输出与参考数据的误差：

```python
MAE = mean(|output - reference|)
```

验证通过条件：`MAE < 1e-2`

对于我们的实现，由于使用相同的算法生成参考数据，MAE = 0（位精确）。

## 扩展支持

### 不同的矩阵大小

```bash
# 小矩阵测试
python utils/generate_data.py 64 64 64

# 大矩阵测试
python utils/generate_data.py 4096 4096 64
```

### 不同的 blocksize

支持 blocksize = 64 或 128：

```bash
python utils/generate_data.py 1024 1024 128
```

### 不同的 GPU 架构

修改 CMakeLists.txt 中的 `CMAKE_CUDA_ARCHITECTURES`：

```cmake
# Volta (V100)
set(CMAKE_CUDA_ARCHITECTURES 70)

# Turing (T4)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Ampere (A100)
set(CMAKE_CUDA_ARCHITECTURES 80)

# Ampere (RTX 30xx)
set(CMAKE_CUDA_ARCHITECTURES 86)
```

## 已知限制

1. **数据类型**：当前输出为 FP16，BF16 支持需要 sm_80+
2. **输入格式**：要求使用本项目的数据生成器
3. **批处理**：当前仅支持单个矩阵

## 故障排除

### 编译错误

- 确保 CUDA Toolkit 已安装（nvcc, cuda_runtime.h）
- 检查 CMake 版本 >= 3.14
- 确保 C++17 支持

### 运行时错误

- 检查 GPU 可用性：`nvidia-smi`
- 验证数据文件存在：`ls -lh data/`
- 查看 CUDA 错误信息（程序会输出详细错误）

### MAE 较大

- 确保使用正确的数据生成器
- 检查 NF4_LUT 值是否正确
- 验证解量化公式实现

## 开发者注意事项

### 修改 Kernel

编辑 `src/nf4_kernel.cu`，注意：
- 保持 4-bit 解包逻辑正确（低位/高位）
- 确保边界检查防止越界访问
- 使用 `CHECK_KERNEL()` 检查启动错误

### 优化建议

1. **使用 Shared Memory**：缓存 code2 和 absmax2
2. **Warp-level 优化**：使用 `__shfl` 指令
3. **多 Stream**：并行处理多个矩阵
4. **Tensor Core**：如果输出需要进一步计算

## 版权和许可

本项目用于学习和研究目的。

## 联系方式

如有问题或建议，请通过 GitHub Issues 反馈。
