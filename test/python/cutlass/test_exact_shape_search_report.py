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
                            "dtype_a": "bf16",
                            "dtype_b": "bf16",
                            "dtype_c": "f32",
                            "dtype_d": "f32",
                            "dtype_acc": "f32",
                        },
                        "KernelSlow": {
                            "layout": "rcr",
                            "runner": "benchmark",
                            "scheduler_family": "Gemm",
                            "decomposition_mode": "Gemm",
                            "streamk_mode": "",
                            "reduction_mode": "None",
                            "tile_m": 256,
                            "tile_n": 128,
                            "tile_k": 64,
                            "sg_m": 8,
                            "sg_n": 4,
                            "stages": 2,
                            "split_k": 1,
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

            with (report_dir / "8192_384_3584" / "ranked_by_total_runtime.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["kernel"] for row in rows], ["KernelFast", "KernelSlow"])
            self.assertEqual(rows[0]["latency_source"], "derived_from_tflops")
            self.assertNotEqual(rows[0]["avg_runtime_ms"], "")
            self.assertNotEqual(rows[0]["total_runtime_ms"], "")

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
