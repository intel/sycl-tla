#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_SCRIPT = REPO_ROOT / "tools" / "exact_shape_search_report.py"
GEN_MAIN_SCRIPT = REPO_ROOT / "tools" / "gen_main.py"


class TestExactShapeSearchReport(unittest.TestCase):
    def test_report_derives_latency_for_legacy_csv_and_writes_rankings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            result_dir = run_dir / "results" / "8192_384_3584"
            report_dir = run_dir / "reports"
            result_dir.mkdir(parents=True)

            (run_dir / "requested_shapes.json").write_text(
                json.dumps({"shapes": [{"m": 8192, "n": 384, "k": 3584}]}, indent=2) + "\n",
                encoding="utf-8",
            )
            (run_dir / "manifest.json").write_text(
                json.dumps({"total_kernels": 2, "batch_size": 1, "batch_count": 2, "gpu_count": 1}, indent=2) + "\n",
                encoding="utf-8",
            )
            (run_dir / "run_meta.txt").write_text(
                "\n".join(
                    [
                        "git_head=deadbeef",
                        "repo_root=/tmp/fake-repo",
                        "kernel_catalog_source=layered_bmg_scheduler_expanded",
                        "benchmark_input_mode=rotating_vram_pool",
                        "benchmark_stride_policy=fixed_4_1_0",
                        "perf_env_ONEAPI_DEVICE_SELECTOR=level_zero:gpu",
                        "perf_env_SYCL_PROGRAM_COMPILE_OPTIONS=-ze-opt-large-register-file -gline-tables-only",
                        "perf_env_IGC_VectorAliasBBThreshold=10000",
                        "perf_env_IGC_ExtraOCLOptions=-cl-intel-256-GRF-per-thread",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            (run_dir / "benchmark_config.json").write_text(
                json.dumps(
                    {
                        "input_mode": "rotating_vram_pool",
                        "stride_policy": "fixed_4_1_0",
                        "input_pool_target_bytes": 1073741824,
                        "warmup_iters": 50,
                        "measure_iters": 100,
                        "fixed_vram_input": False,
                        "phase_timing_enabled": False,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            (run_dir / "synced_sources.json").write_text(
                json.dumps(
                    [
                        {
                            "path": "benchmarks/gemm/benchmark_runner.hpp",
                            "exists": True,
                            "sha256": "abc123",
                        }
                    ],
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            (run_dir / "kernel_metadata.json").write_text(
                json.dumps(
                    {
                        "KernelFast": {
                            "layout": "rrr",
                            "runner": "benchmark",
                            "scheduler_family": "Gemm",
                            "decomposition_mode": "Gemm",
                            "streamk_mode": "",
                            "reduction_mode": "None",
                            "tile_m": 128,
                            "tile_n": 128,
                            "tile_k": 64,
                            "sg_m": 4,
                            "sg_n": 8,
                            "stages": 2,
                            "split_k": 1,
                            "kernel_schedule": "KernelXe",
                            "tile_scheduler": "Gemm",
                            "source": "exhaustive_regular_gemm_catalog",
                            "allowed_runtime_sweeps": ["m", "n", "k"],
                            "dtype_a": "bf16",
                            "dtype_b": "bf16",
                            "dtype_c": "f32",
                            "dtype_d": "f32",
                            "dtype_acc": "f32",
                        },
                        "KernelSlow": {
                            "layout": "rcr",
                            "runner": "benchmark",
                            "scheduler_family": "SplitK",
                            "decomposition_mode": "SplitK",
                            "streamk_mode": "splitk",
                            "reduction_mode": "Workspace",
                            "tile_m": 256,
                            "tile_n": 128,
                            "tile_k": 64,
                            "sg_m": 8,
                            "sg_n": 4,
                            "stages": 2,
                            "split_k": 1,
                            "kernel_schedule": "KernelXeCooperative",
                            "tile_scheduler": "StreamKScheduler",
                            "source": "exhaustive_streamk_catalog",
                            "allowed_runtime_sweeps": ["m", "n", "k", "split_k_slices"],
                            "dtype_a": "bf16",
                            "dtype_b": "bf16",
                            "dtype_c": "f32",
                            "dtype_d": "f32",
                            "dtype_acc": "f32",
                        },
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            with (result_dir / "batch_0000_gpu0.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["kernel", "tflops", "status", "gpu", "m", "n", "k"])
                writer.writeheader()
                writer.writerow(
                    {
                        "kernel": "KernelFast",
                        "tflops": "140.0",
                        "status": "OK",
                        "gpu": "0",
                        "m": "8192",
                        "n": "384",
                        "k": "3584",
                    }
                )
                writer.writerow(
                    {
                        "kernel": "KernelSlow",
                        "tflops": "70.0",
                        "status": "OK",
                        "gpu": "0",
                        "m": "8192",
                        "n": "384",
                        "k": "3584",
                    }
                )
                writer.writerow(
                    {
                        "kernel": "KernelTimeout",
                        "tflops": "0",
                        "status": "TIMEOUT",
                        "gpu": "0",
                        "m": "8192",
                        "n": "384",
                        "k": "3584",
                    }
                )

            subprocess.run(
                [
                    sys.executable,
                    str(REPORT_SCRIPT),
                    "--run-dir",
                    str(run_dir),
                    "--shape-tag",
                    "8192_384_3584",
                    "--output-dir",
                    str(report_dir),
                ],
                check=True,
            )

            summary = json.loads((report_dir / "8192_384_3584" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["row_count"], 3)
            self.assertEqual(summary["ok_row_count"], 2)
            self.assertEqual(summary["fastest5_latency"][0]["kernel"], "KernelFast")
            self.assertEqual(summary["fastest5_rcr_latency"][0]["kernel"], "KernelSlow")
            self.assertEqual(summary["top5"][0]["kernel"], "KernelFast")
            self.assertEqual(summary["top5"][0]["latency_source"], "derived_from_tflops")
            self.assertEqual(summary["fastest5_latency"][0]["measure_iters"], "100")
            self.assertIn("total_runtime_ms", summary["latency_stats"])
            self.assertIn("kernel_schedule", summary["merged_fields"])
            self.assertEqual(summary["run_meta"]["git_head"], "deadbeef")
            self.assertEqual(summary["manifest"]["batch_count"], 2)
            self.assertEqual(summary["benchmark_config"]["stride_policy"], "fixed_4_1_0")
            self.assertEqual(summary["synced_sources"][0]["sha256"], "abc123")
            self.assertEqual(summary["search_limitations"][0]["constraint"], "runtime split_k_slices <= 1")
            self.assertTrue((report_dir / "8192_384_3584" / "top1_filter.txt").exists())
            self.assertTrue((report_dir / "8192_384_3584" / "top1_repro.cfg").exists())
            self.assertTrue((report_dir / "8192_384_3584" / "top1_repro.sh").exists())
            self.assertIn("top1", summary["repro_artifacts"])

            with (report_dir / "8192_384_3584" / "ranked_by_total_runtime.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["kernel"] for row in rows], ["KernelFast", "KernelSlow"])
            self.assertEqual(rows[0]["latency_source"], "derived_from_tflops")
            self.assertNotEqual(rows[0]["avg_runtime_ms"], "")
            self.assertNotEqual(rows[0]["total_runtime_ms"], "")
            self.assertEqual(rows[0]["kernel_schedule"], "KernelXe")
            self.assertEqual(rows[0]["allowed_runtime_sweeps"], "[\"m\", \"n\", \"k\"]")

            top1_filter = (report_dir / "8192_384_3584" / "top1_filter.txt").read_text(encoding="utf-8")
            self.assertEqual(top1_filter.strip(), "^KernelFast$")
            top1_cfg = (report_dir / "8192_384_3584" / "top1_repro.cfg").read_text(encoding="utf-8")
            self.assertIn("KernelFast", top1_cfg)
            self.assertIn("--alpha=1", top1_cfg)
            top5_cfg = (report_dir / "8192_384_3584" / "top5_repro.cfg").read_text(encoding="utf-8")
            self.assertIn("KernelSlow", top5_cfg)
            self.assertIn("--split_k_slices=1", top5_cfg)
            top1_script = (report_dir / "8192_384_3584" / "top1_repro.sh").read_text(encoding="utf-8")
            self.assertIn("--config_file=\"$CONFIG_FILE\"", top1_script)
            self.assertIn("KERNEL_FILTER_FILE=\"$FILTER_FILE\"", top1_script)
            self.assertIn('PATH="/opt/intel/oneapi/compiler/2025.3/bin:$PATH"', top1_script)
            self.assertIn('-DCMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER:-/opt/intel/oneapi/compiler/2025.3/bin/icpx}"', top1_script)
            self.assertIn('SHARED_DEPS_BUILD="${SHARED_DEPS_BUILD:-$RUN_DIR/workers/gpu0/build}"', top1_script)
            self.assertIn('ln -sfn "$SHARED_DEPS_BUILD/_deps/googlebenchmark-build" "$BUILD_DIR/_deps/googlebenchmark-build"', top1_script)
            self.assertIn("ZE_AFFINITY_MASK", top1_script)

    def test_gen_main_emits_latency_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "manifest.txt"
            output = Path(tmpdir) / "main.cpp"
            manifest.write_text("KernelFast\n", encoding="utf-8")

            subprocess.run([sys.executable, str(GEN_MAIN_SCRIPT), str(manifest), str(output)], check=True)

            text = output.read_text(encoding="utf-8")
            self.assertIn("avg_runtime_ms", text)
            self.assertIn("total_runtime_ms", text)
            self.assertIn("measure_iters", text)
            self.assertIn("warmup_iters", text)
            self.assertIn("opts.split_k_slices = 0", text)
            self.assertIn("cmd.get_cmd_line_argument(\"l\", opts.l, 1)", text)


if __name__ == "__main__":
    unittest.main()
