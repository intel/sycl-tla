#!/usr/bin/env python3
"""
Generate a per-batch slim benchmarks_sycl.hpp containing only the kernel types
needed for the current batch.  This eliminates template instantiation bloat
and brings SYCL compile time from 30+ min to ~2 min per batch.

Usage:
  python3 tools/slim_benchmarks.py --kernels BmgGemmBF16BF16FP32_RCR_6,BmgGemmBF16BF16FP32_RRR_6
"""

import re, sys, argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FULL_HPP = REPO / "benchmarks/gemm/benchmarks_sycl.hpp"
DEF_FILE = REPO / "benchmarks/gemm/bmg_gemm_source_tile_sg.def"

def parse_source_tiles():
    tiles = []
    with open(DEF_FILE) as f:
        for line in f:
            m = re.match(r"BMG_SOURCE_GEMM_TILE_SG\((\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)", line.strip())
            if m:
                tiles.append(tuple(map(int, m.groups())))
    return tiles

def extract_kernel_families(kernel_list):
    """Categorize kernels into families to know which template blocks to include."""
    families = set()
    for kn in kernel_list:
        if "_RRR_" in kn or "RRR_TileShape" in kn:
            families.add("BF16_RRR")
        elif "_RCR_" in kn:
            if "StreamK" in kn or "DataParallel" in kn or "SplitK" in kn:
                families.add("BF16_RCR_StreamK")
            elif "Gemm_" in kn and "SG" in kn:
                families.add("BF16_RCR_SOURCE")
            elif "_RCR_" in kn and re.match(r".*_RCR_\d+$", kn):
                families.add("BF16_RCR_HANDTUNED")
            else:
                families.add("BF16_RCR_BASIC")
    return families

def generate(kernels, output_path):
    with open(FULL_HPP) as f:
        full = f.read()

    families = extract_kernel_families(kernels)
    
    # Build slim header
    lines = []
    lines.append("#pragma once")
    lines.append('#include "gemm_configuration_sycl.hpp"')
    lines.append("")
    lines.append("using Scheduler = cutlass::gemm::device::Scheduler;")
    lines.append("")

    # Always include MMAAtom for BF16
    lines.append('using MMAAtom = MMA_Atom<XE_DPAS_TT<8, float, cute::bfloat16_t>>;')
    lines.append("")

    # Extract the Gemm_Bench template for BF16 RCR
    if any(f in families for f in ("BF16_RCR_BASIC", "BF16_RCR_HANDTUNED", "BF16_RCR_SOURCE", "BF16_RCR_StreamK")):
        lines.append("// Gemm_Bench_BF16FP32_RCR template")
        for i, line in enumerate(full.split("\n")):
            if "using Gemm_Bench_BF16FP32_RCR = cutlass::gemm::device::GemmConfiguration<" in line:
                # Include template up to the closing ">;"
                for j in range(i, i+15):
                    l = full.split("\n")[j]
                    lines.append(l)
                    if l.strip() == "PipelineStages>;" or l.strip().endswith("PipelineStages>;"):
                        break
                break

    # Extract the Gemm_Bench template for BF16 RRR
    if "BF16_RRR" in families:
        for i, line in enumerate(full.split("\n")):
            if "using Gemm_Bench_BF16FP32_RRR = cutlass::gemm::device::GemmConfiguration<" in line:
                for j in range(i, i+15):
                    l = full.split("\n")[j]
                    lines.append(l)
                    if l.strip() == "PipelineStages>;" or l.strip().endswith("PipelineStages>;"):
                        break
                break

    # Extract the StreamK template for BF16 RCR
    if "BF16_RCR_StreamK" in families:
        # Find StreamK tile type + config template
        found_tile = False
        for line in full.split("\n"):
            if "BmgGemm_BF16BF16FP32_StreamK_Tile =" in line:
                found_tile = True
            if found_tile:
                lines.append(line)
                if "KernelXeCooperative>;" in line:
                    found_tile = False
                    break

    # Add hand-tuned kernel definitions
    hand_tuned = sorted([k for k in kernels if re.match(r".*_RCR_\d+$", k)])
    for kn in hand_tuned:
        for i, line in enumerate(full.split("\n")):
            if f"using {kn} = " in line and "Gemm_Bench" in line:
                lines.append(line)
                lines.append(full.split("\n")[i+1])  # CUTLASS_CREATE
                break

    # Add RRR kernel definitions
    rrr_kernels = sorted([k for k in kernels if "RRR" in k])
    for kn in rrr_kernels:
        for i, line in enumerate(full.split("\n")):
            if f"using {kn} = " in line or f"CUTLASS_CREATE_GEMM_BENCHMARK({kn})" in line:
                lines.append(line)
            elif f"using BmgGemmBF16BF16FP32_{kn.replace('BmgGemmBF16BF16FP32_', '')}_Tile" in line or f"// RRR" in line and "TiledMMAHelper" in line:
                lines.append(line)

    # Add register function
    lines.append("")
    lines.append("static void register_gemm_benchmarks() {")
    for kn in sorted(set(kernels)):
        lines.append(f"  CUTLASS_BENCHMARK({kn});")
    lines.append("}")
    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"Generated slim benchmarks_sycl.hpp: {len(kernels)} kernels, {len(lines)} lines")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--kernels", required=True, help="Comma-separated kernel names")
    p.add_argument("--output", default="/tmp/slim_benchmarks_sycl.hpp")
    args = p.parse_args()
    kernel_list = [k.strip() for k in args.kernels.split(",") if k.strip()]
    generate(kernel_list, args.output)
