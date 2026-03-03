"""
Data generator for NF4 dequantization testing.

Uses bitsandbytes to perform real NF4 quantization, and saves:
  - weights.bin     : quantized parameters in binary format for the CUDA kernel
  - reference.bin   : dequantized output from bitsandbytes (ground truth for MAE validation)
  - params.txt      : config parameters

Binary format for weights.bin:
  [Header: 20 bytes]
    int64  num_rows
    int64  num_cols
    int32  blocksize

  [Data]
    uint8[num_rows*num_cols/2]  packed_weights   (2 NF4 indices per byte)
    uint8[num_blocks]           absmax_q          (level-1 quantized scale, uint8)
    fp16[num_groups]            absmax2           (level-2 scale)
    fp16[256]                   code2             (codebook for absmax_q)
    float32                     offset
"""

import os
import struct
import argparse

import numpy as np
import torch
from bitsandbytes.functional import dequantize_4bit, quantize_4bit


def generate_nf4_test_data(num_rows=1024, num_cols=1024, blocksize=64, output_dir="data", dtype="fp16"):
    os.makedirs(output_dir, exist_ok=True)

    total_elements = num_rows * num_cols
    num_blocks = (total_elements + blocksize - 1) // blocksize
    num_groups = (num_blocks + 255) // 256

    print(f"Generating NF4 test data (via bitsandbytes):")
    print(f"  Matrix: {num_rows} x {num_cols}, blocksize={blocksize}")
    print(f"  Num blocks: {num_blocks}, num groups: {num_groups}")

    # ------------------------------------------------------------------
    # 1. quantize with NF4 via bitsandbytes quantize_4bit
    #    compress_statistics=True enables nested/double quantization:
    #      state.absmax           (uint8[num_blocks])  – quantized block scale
    #      state.nested_quant_map (float32[256])        – codebook for absmax
    #      state.nested_absmax    (float32[num_groups]) – group-level scale
    #      state.nested_offset    (float32)             – added during absmax reconstruction
    #    Reconstruction: block_scale = nested_quant_map[absmax[b]] * nested_absmax[b//256] + nested_offset
    # ------------------------------------------------------------------
    weights = torch.randn(num_rows, num_cols, dtype=torch.float32)
    quantized_data, state = quantize_4bit(
        weights, blocksize=blocksize, quant_type="nf4", compress_statistics=True
    )
    d = state.as_dict()

    # packed_weights: uint8 array, 2 NF4 indices per byte → size = total_elements / 2
    packed_weights = quantized_data.cpu().numpy().flatten()
    assert packed_weights.nbytes == total_elements // 2, \
        f"Expected {total_elements // 2} bytes, got {packed_weights.nbytes}"

    # absmax_q: level-1 quantized scale indices per block (uint8)
    absmax_q = d["absmax"].cpu().numpy().astype(np.uint8)

    # code2: nested quantization codebook (fp16[256])
    code2  = d["nested_quant_map"].cpu().numpy().astype(np.float16)
    # absmax2: group-level scale for nested quantization (fp16[num_groups])
    absmax2 = d["nested_absmax"].cpu().numpy().astype(np.float16)
    # offset: added to reconstructed block scale (from nested_offset)
    offset = float(d["nested_offset"])

    print(f"\nbitsandbytes nested state:")
    print(f"  packed_weights: {packed_weights.shape}, dtype={packed_weights.dtype}")
    print(f"  absmax_q:       {absmax_q.shape}, dtype={absmax_q.dtype}, range=[{absmax_q.min()}, {absmax_q.max()}]")
    print(f"  code2:          {code2.shape}, dtype={code2.dtype}")
    print(f"  absmax2:        {absmax2.shape}, dtype={absmax2.dtype}")
    print(f"  nested_offset:  {offset}")

    # ------------------------------------------------------------------
    # 2. Write weights.bin
    # ------------------------------------------------------------------
    weights_file = os.path.join(output_dir, "weights.bin")
    with open(weights_file, "wb") as f:
        f.write(struct.pack("<q", num_rows))
        f.write(struct.pack("<q", num_cols))
        f.write(struct.pack("<i", blocksize))
        f.write(packed_weights.tobytes())
        f.write(absmax_q.tobytes())
        f.write(absmax2.tobytes())
        f.write(code2.tobytes())
        f.write(struct.pack("<f", offset))

    expected_size = 20 + packed_weights.nbytes + absmax_q.nbytes + absmax2.nbytes + code2.nbytes + 4
    actual_size   = os.path.getsize(weights_file)
    print(f"\nWrote {weights_file}")
    print(f"  Expected size: {expected_size} bytes, actual: {actual_size} bytes  {'✓' if expected_size == actual_size else '✗ MISMATCH'}")

    # ------------------------------------------------------------------
    # 3. Reference: bitsandbytes dequantize_4bit (ground truth)
    # ------------------------------------------------------------------
    dequantized = dequantize_4bit(quantized_data, state, quant_type="nf4")  # fp32 tensor

    # numpy has no native bfloat16 type, so for bf16 we use torch to cast then
    # reinterpret the underlying bits as int16 to get raw bytes.
    if dtype == "bf16":
        ref_bytes = dequantized.to(torch.bfloat16).view(torch.int16).cpu().numpy().tobytes()
    else:
        ref_bytes = dequantized.cpu().numpy().astype(np.float16).tobytes()

    ref_file = os.path.join(output_dir, "reference.bin")
    with open(ref_file, "wb") as f:
        f.write(ref_bytes)
    print(f"Wrote {ref_file}  ({len(ref_bytes)} bytes, {dtype})")

    # Sanity-check bitsandbytes own MAE vs original
    bn_mae = float(torch.abs(weights - dequantized).mean())
    print(f"\nbitsandbytes internal MAE (fp32 weights vs nf4-dequant): {bn_mae:.6f}")

    # ------------------------------------------------------------------
    # 4. params.txt
    # ------------------------------------------------------------------
    params_file = os.path.join(output_dir, "params.txt")
    with open(params_file, "w") as f:
        f.write(f"blocksize = {blocksize}\n")
        f.write(f'compute_type = "{dtype}"\n')
        f.write(f'target_gpu = "T4"\n')
    print(f"Wrote {params_file}")
    print("\nData generation complete!")


def load_args():
    parser = argparse.ArgumentParser(description="Generate NF4 test data using bitsandbytes")
    parser.add_argument("rows",      type=int, nargs="?", default=1024)
    parser.add_argument("cols",      type=int, nargs="?", default=1024)
    parser.add_argument("blocksize", type=int, nargs="?", default=64)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16",
                        help="Output dtype for reference.bin and params.txt compute_type (default: fp16)")
    return parser.parse_args()


if __name__ == "__main__":
    args = load_args()
    generate_nf4_test_data(args.rows, args.cols, args.blocksize, args.output_dir, args.dtype)
