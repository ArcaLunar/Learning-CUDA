"""
Data generator for NF4 dequantization - proper format version.
Creates test data with correct binary format.
"""

import struct
import numpy as np
import sys
import argparse


def generate_nf4_test_data(num_rows=1024, num_cols=1024, blocksize=64):
    """Generate random NF4-like quantized data with proper format."""

    total_elements = num_rows * num_cols
    num_blocks = (total_elements + blocksize - 1) // blocksize
    num_groups = (num_blocks + 255) // 256

    print(f"Generating NF4 test data:")
    print(f"  Matrix: {num_rows} x {num_cols}")
    print(f"  Total elements: {total_elements}")
    print(f"  Blocksize: {blocksize}")
    print(f"  Num blocks: {num_blocks}")
    print(f"  Num groups: {num_groups}")

    # Generate random packed weights (4-bit indices, 2 per byte)
    packed_weights = np.random.randint(0, 256, size=total_elements // 2, dtype=np.uint8)

    # Generate random scale factors
    # absmax_q: quantized scale factors (uint8, values 0-255)
    absmax_q = np.random.randint(50, 200, size=num_blocks, dtype=np.uint8)

    # absmax2: group-level scale factors (float16, reasonable values)
    absmax2 = np.random.uniform(0.5, 2.0, size=num_groups).astype(np.float16)

    # code2: codebook mapping uint8 to float16 scale values
    code2 = np.linspace(0.001, 0.1, 256, dtype=np.float16)

    # offset: usually 0
    offset = 0.0

    # Write weights.bin
    weights_file = "data/weights.bin"
    with open(weights_file, "wb") as f:
        # Header
        f.write(struct.pack("<q", num_rows))
        f.write(struct.pack("<q", num_cols))
        f.write(struct.pack("<i", blocksize))

        # Data
        f.write(packed_weights.tobytes())
        f.write(absmax_q.tobytes())
        f.write(absmax2.tobytes())
        f.write(code2.tobytes())
        f.write(struct.pack("<f", offset))

    file_size = (
        20 + len(packed_weights) + len(absmax_q) + len(absmax2) * 2 + len(code2) * 2 + 4
    )
    print(f"\nWrote {weights_file}")
    print(f"Expected size: {file_size} bytes")

    import os

    actual_size = os.path.getsize(weights_file)
    print(f"Actual size: {actual_size} bytes")
    print(f"Match: {file_size == actual_size}")

    # Generate reference by manually dequantizing
    # NF4 lookup table
    nf4_lut = np.array(
        [
            -1.0,
            -0.6961928009986877,
            -0.5250730514526367,
            -0.39491748809814453,
            -0.28444138169288635,
            -0.18477343022823334,
            -0.09105003625154495,
            0.0,
            0.07958029955625534,
            0.16093020141124725,
            0.24611230194568634,
            0.33791524171829224,
            0.44070982933044434,
            0.5626170039176941,
            0.7229568362236023,
            1.0,
        ],
        dtype=np.float32,
    )

    reference = np.zeros((num_rows, num_cols), dtype=np.float16)

    for i in range(total_elements):
        # Get 4-bit index
        byte_idx = i // 2
        if i % 2 == 0:
            idx = packed_weights[byte_idx] & 0x0F  # Low 4 bits
        else:
            idx = (packed_weights[byte_idx] >> 4) & 0x0F  # High 4 bits

        # Get block and group indices
        block_idx = i // blocksize
        group_idx = block_idx // 256

        # Dequantize
        base_val = nf4_lut[idx]
        scale1 = code2[absmax_q[block_idx]]
        scale2 = absmax2[group_idx]
        value = base_val * scale1 * scale2 + offset

        reference[i // num_cols, i % num_cols] = value

    # Write reference
    ref_file = "data/reference.bin"
    with open(ref_file, "wb") as f:
        f.write(reference.tobytes())

    print(f"\nWrote {ref_file}")
    print(f"Size: {reference.nbytes} bytes")

    # Update params.txt
    params_file = "data/params.txt"
    with open(params_file, "w") as f:
        f.write(f"blocksize = {blocksize}\n")
        f.write(f'compute_type = "bf16"\n')
        f.write(f'target_gpu = "T4"\n')

    print(f"\nUpdated {params_file}")
    print("\nData generation complete!")


def load_args():
    parser = argparse.ArgumentParser(description="Generate NF4 test data")
    parser.add_argument(
        "rows",
        type=int,
        nargs="?",
        default=1024,
        help="Number of rows",
    )
    parser.add_argument(
        "cols",
        type=int,
        nargs="?",
        default=1024,
        help="Number of columns",
    )
    parser.add_argument(
        "blocksize",
        type=int,
        nargs="?",
        default=64,
        help="Block size for quantization",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = load_args()
    rows = args.rows
    cols = args.cols
    blocksize = args.blocksize

    generate_nf4_test_data(rows, cols, blocksize)
