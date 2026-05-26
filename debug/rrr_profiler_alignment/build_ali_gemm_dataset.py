#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import argparse
import importlib.util
from pathlib import Path


def load_profiler_module():
    module_path = Path(__file__).with_name("intel_gemm_profiler.py")
    spec = importlib.util.spec_from_file_location("intel_gemm_profiler", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main():
    profiler = load_profiler_module()
    parser = argparse.ArgumentParser(description="Build supported shape/reference docs from the ali GEMM Excel dataset.")
    parser.add_argument("--xlsx", required=True, help="Path to ali_gemm_perf_v0.1.xlsx")
    parser.add_argument("--output-dir", required=True, help="Directory for generated JSON files.")
    parser.add_argument("--layouts", default="rcr,rrr", help="Comma-separated layout tags to assign to generated shapes (e.g., rcr,rrr).")
    args = parser.parse_args()

    output_dir = profiler.ensure_dir(Path(args.output_dir).resolve())
    layout_list = tuple(l.strip() for l in args.layouts.split(","))
    shapes_doc, reference_doc = profiler.build_ali_gemm_docs(args.xlsx, layouts=layout_list)
    profiler.write_json(output_dir / "ali_gemm_shapes.json", shapes_doc)
    profiler.write_json(output_dir / "ali_gemm_reference.json", reference_doc)
    print(
        {
            "shapes_json": str(output_dir / "ali_gemm_shapes.json"),
            "reference_json": str(output_dir / "ali_gemm_reference.json"),
            "shape_count": len(shapes_doc["shapes"]),
            "reference_entries": len(reference_doc["entries"]),
        }
    )


if __name__ == "__main__":
    main()
