#!/usr/bin/env python3
"""
Generate per-batch minimal benchmarks_sycl.hpp containing ONLY the kernel types
needed for this batch.  Reduces SYCL compile time from 30+ min to ~2 min.

The output file includes:
  - Core includes + MMAAtom definition
  - Gemm_Bench_BF16FP32_RCR template (compact)
  - Gemm_Bench_BF16FP32_RRR template (compact)  
  - Only the kernel type definitions + CUTLASS_CREATE lines for requested kernels
  - Minimal register_gemm_benchmarks()
"""
import re, sys, argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FULL_HPP = REPO / "benchmarks/gemm/benchmarks_sycl.hpp"

FULL_CACHE = REPO / "benchmarks/gemm/benchmarks_sycl.hpp.cache"
def parse_full():
    if not FULL_CACHE.exists():
        import shutil
        shutil.copy2(REPO / "benchmarks/gemm/benchmarks_sycl.hpp", FULL_CACHE)
    with open(FULL_CACHE) as f:
        return f.readlines()

def find_block(lines, kernel_name):
    """Return all lines needed to define a kernel type (type def + CREATE + exact deps)."""
    needed = []
    for i, line in enumerate(lines):
        if f"using {kernel_name} = " in line:
            # Find EXACT tile dependencies referenced in this line
            deps = set()
            for tile_name in re.findall(r'(BmgTile_\w+|BmgGemm_\w+_Tile\w*|BmgF16Tile_\w+)', line):
                deps.add(tile_name)
            for tile_name in re.findall(r'(BmgGemm\w+_TileShape\w+)', line):
                deps.add(tile_name)
            # Find each dependency definition (exact name match)
            for dep in deps:
                for dep_line in lines:
                    # Exact match: "using DepName =" not "using DepName_other ="
                    if re.match(rf'\s*using\s+{re.escape(dep)}\s*=', dep_line):
                        needed.append(dep_line)
                        break
            needed.append(line)
            if i+1 < len(lines) and "CUTLASS_CREATE_GEMM_BENCHMARK" in lines[i+1]:
                needed.append(lines[i+1])
    return needed

def make_mini(kernels, output_path):
    lines = parse_full()
    
    # Collect all type definitions
    blocks = []
    for k in sorted(set(kernels)):
        block = find_block(lines, k)
        blocks.extend(block)
    
    # Deduplicate
    seen = set()
    uniq_blocks = []
    for b in blocks:
        h = b.strip()
        if h not in seen:
            seen.add(h)
            uniq_blocks.append(b)
    
    with open(output_path, "w") as f:
        f.write("""#pragma once
#include "../../../benchmarks/gemm/gemm_configuration_sycl.hpp"
using Scheduler = cutlass::gemm::device::Scheduler;

// MMA atom (modern — matches example 00)
using MMAAtom = MMA_Atom<XE_DPAS_TT<8, float, cute::bfloat16_t>>;

// Gemm_Bench_BF16FP32_RCR template (compact — for RCR types)
template <typename TileShape, typename Tiler, typename GmemCA, typename GmemCB, int PS=2>
using Gemm_Bench_BF16FP32_RCR = cutlass::gemm::device::GemmConfiguration<
    cutlass::arch::IntelXe,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor, float,
    TileShape, Scheduler::Gemm, Tiler,
    GmemCA, GmemCB,
    cutlass::epilogue::fusion::LinearCombination<float,float,float,float,cutlass::FloatRoundStyle::round_to_nearest>,
    PS>;

// Gemm_Bench_BF16FP32_RRR template (compact — for RRR types)
template <typename TileShape, typename Tiler, typename GmemCA, typename GmemCB, int PS=2>
using Gemm_Bench_BF16FP32_RRR = cutlass::gemm::device::GemmConfiguration<
    cutlass::arch::IntelXe,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    cutlass::bfloat16_t, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor, float,
    TileShape, Scheduler::Gemm, Tiler,
    GmemCA, GmemCB,
    cutlass::epilogue::fusion::LinearCombination<float,float,float,float,cutlass::FloatRoundStyle::round_to_nearest>,
    PS>;

""")
        f.writelines(uniq_blocks)
        f.write("""
static void register_gemm_benchmarks() {
""")
        for k in sorted(set(kernels)):
            f.write(f"  CUTLASS_BENCHMARK({k});\n")
        f.write("}\n")
    
    # Overwrite source file for correct #include resolution
    src_path = REPO / "benchmarks/gemm/benchmarks_sycl.hpp"
    import shutil
    shutil.copy2(output_path, src_path)
    print(f"Generated {output_path}: {len(kernels)} kernels, {len(uniq_blocks) + 60} lines → overwrote {src_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    with open(args.manifest) as f:
        kernels = [l.strip() for l in f if l.strip()]
    make_mini(kernels, args.output)


def restore_original():
    src = REPO / "benchmarks/gemm/benchmarks_sycl.hpp"
    bak = REPO / "benchmarks/gemm/benchmarks_sycl.hpp.bak"
    if bak.exists():
        import shutil
        shutil.move(str(bak), str(src))
        print(f"Restored original benchmarks_sycl.hpp from backup")
