import torch
import numpy as np
from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise
import struct
import os
import argparse


def generate_nf4_data(num_rows, num_cols, blocksize, output_dir="data"):
    """
    Generate test data for NF4 quantization.

    Args:
        num_rows: number of rows in the weight matrix
        num_cols: number of columns in the weight matrix
        blocksize: size of each block (64 or 128)
        output_dir: directory to save the generated files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Generate random weight matrix
    print(f"Random weight matrix of size [{num_rows}, {num_cols}]")
    weights = torch.randn(num_rows, num_cols, dtype=torch.float32)

    # NF4 quant with bitsandbytes lib
    print(f"NF4 quantization with blocksize={blocksize}...")
    quantized_data, state = quantize_blockwise(weights, blocksize=blocksize, code=None)

    # extract quantization parameters
    packed_weights = quantized_data.cpu().numpy()  # uint8 array, 1*uint8 = 2*NF4
    absmax_q = state.absmax.cpu().numpy()  # Level 1 scaling factors (uint8)

    # Level 2 scaling factors and codebook (fp16)
    if hasattr(state, "state2") and state.state2 is not None:
        absmax2 = state.state2.absmax.cpu().numpy()  # level 2 scaling factors (float16)
        code2 = state.state2.code.cpu().numpy()  # codebook (float16[256])
    else:
        # default value if state2 is not provided (e.g. for blocksize=128)
        num_blocks = (num_rows * num_cols + blocksize - 1) // blocksize
        num_groups = (num_blocks + 255) // 256  # 256 blocks per group for level 2
        absmax2 = np.ones(num_groups, dtype=np.float16)
        code2 = np.linspace(-1.0, 1.0, 256, dtype=np.float16)

    # offset (float32)
    if hasattr(state, "offset") and state.offset is not None:
        offset = float(state.offset)
    else:
        offset = 0.0

    # dump data to binary file
    weight_file = os.path.join(output_dir, "weights.bin")
    print(f"Saving weights to: {weight_file}")

    with open(weight_file, "wb") as f:
        # header
        f.write(struct.pack("<q", num_rows))  # int64
        f.write(struct.pack("<q", num_cols))  # int64
        f.write(struct.pack("<i", blocksize))  # int32

        # data
        f.write(packed_weights.tobytes())  # uint8 数组

        # absmax_q => uint8
        if absmax_q.dtype != np.uint8:
            absmax_q_u8 = np.clip((absmax_q * 255.0).astype(np.uint8), 0, 255)
        else:
            absmax_q_u8 = absmax_q
        f.write(absmax_q_u8.tobytes())

        # absmax2 (float16)
        f.write(absmax2.astype(np.float16).tobytes())

        # code2 (float16[256])
        f.write(code2.astype(np.float16).tobytes())

        # offset (float32)
        f.write(struct.pack("<f", float(offset)))

    # dump parameters to text file
    param_file = os.path.join(output_dir, "params.txt")
    print(f"Saving parameters to: {param_file}")

    with open(param_file, "w") as f:
        f.write(f"blocksize = {blocksize}\n")
        f.write(f'compute_type = "bf16"\n')
        f.write(f'target_gpu = "T4"\n')

    # dump reference weights (float16) for validation
    ref_file = os.path.join(output_dir, "reference.bin")
    print(f"Saving reference weights to: {ref_file}")
    weights_fp16 = weights.cpu().numpy().astype(np.float16)
    with open(ref_file, "wb") as f:
        f.write(weights_fp16.tobytes())

    # validate quantization/dequantization
    print("\nValidating quant/dequant...")
    dequantized = dequantize_blockwise(quantized_data, state)
    error = torch.abs(weights - dequantized).mean().item()
    print(f"Mean absolute error: {error:.6f}")

    print(f"\nOutput directory: {output_dir}")
    print(f"  - weights.bin: input data, header + quant weights")
    print(f"  - params.txt: parameters for quantization")
    print(f"  - reference.bin: reference weights in float16 for validation")

    return weights, quantized_data, state


def main():
    parser = argparse.ArgumentParser(description="NF4 Quantization Data Generator")
    parser.add_argument(
        "--rows",
        type=int,
        default=1024,
        help="Number of rows in the weight matrix",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=1024,
        help="Number of columns in the weight matrix",
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=64,
        help="Block size (64 or 128)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("NF4 Quantization Data Generator")
    print("=" * 60)
    print(f"Matrix shape: [{args.rows}, {args.cols}]")
    print(f"Block size: {args.blocksize}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    print()

    generate_nf4_data(args.rows, args.cols, args.blocksize, args.output_dir)


if __name__ == "__main__":
    main()
