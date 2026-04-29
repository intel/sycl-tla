#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#################################################################################################

import importlib.util
import tempfile
import unittest
from pathlib import Path


def load_module():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "test" / "benchmarks" / "intel_gemm_profiler.py"
    spec = importlib.util.spec_from_file_location("intel_gemm_profiler", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


profiler = load_module()


class TestIntelGemmProfiler(unittest.TestCase):
    def test_generate_candidate_space_uses_catalog_level0(self):
        shapes = profiler.default_shapes("bf16")
        constraints = profiler.default_constraints()
        profiles = profiler.default_compiler_profiles()

        candidate_space = profiler.generate_candidate_space(shapes, constraints, profiles)

        self.assertEqual(candidate_space["device_arch"], "bmg")
        self.assertEqual(candidate_space["kernel_catalog"]["catalog_version"], "level0-seed-catalog")
        self.assertEqual(
            candidate_space["search_runtime_schema"]["compile_time_dimensions"][0],
            "dtype_a",
        )
        self.assertEqual(len(candidate_space["candidates"]), 5)
        candidate_ids = {candidate["candidate_id"] for candidate in candidate_space["candidates"]}
        self.assertIn("rcr_bf16bf16f32_tm8_tn128_tk32_sg1x4_st2_sk1", candidate_ids)
        self.assertIn("rcr_bf16bf16f32_tm256_tn256_tk32_sg8x4_st2_sk1", candidate_ids)
        self.assertFalse(any(candidate["split_k"] > 1 for candidate in candidate_space["candidates"]))
        self.assertTrue(all(candidate["filters_applied"][0] == "kernel_catalog" for candidate in candidate_space["candidates"]))
        self.assertTrue(all(candidate["grf_mode"] == 256 for candidate in candidate_space["candidates"]))

        probe_candidate_space = profiler.generate_candidate_space(
            shapes, constraints, profiles, allowed_runners=("benchmark", "streamk_example")
        )
        splitk = next(candidate for candidate in probe_candidate_space["candidates"] if candidate["split_k"] == 2)
        self.assertEqual(splitk["runner"], "streamk_example")
        self.assertEqual(splitk["benchmark_target"], "03_bmg_gemm_streamk")

    def test_build_kernel_catalog_includes_runtime_metadata(self):
        catalog = profiler.build_kernel_catalog(dtypes=["bf16"], allowed_runners=("benchmark", "streamk_example"))

        self.assertEqual(catalog["catalog_version"], "level0-seed-catalog")
        self.assertEqual(catalog["search_runtime_schema"]["runtime_dimensions"][0], "shape_id")
        self.assertEqual(len(catalog["kernels"]), 6)
        splitk = next(entry for entry in catalog["kernels"] if entry["split_k"] == 2)
        self.assertEqual(splitk["instantiation_level"], 0)
        self.assertEqual(splitk["runtime_defaults"]["k_unroll"], 1)
        self.assertIn("barrier_interval", splitk["allowed_runtime_sweeps"])

    def test_build_candidate_build_manifest_splits_compile_and_runtime_fields(self):
        shapes = profiler.default_shapes("bf16")
        constraints = profiler.default_constraints()
        profiles = profiler.default_compiler_profiles()
        candidate_space = profiler.generate_candidate_space(shapes, constraints, profiles)

        manifest = profiler.build_candidate_build_manifest(candidate_space)

        self.assertEqual(manifest["search_runtime_schema"]["microbench_guided_defaults"]["grf_mode"], 256)
        self.assertEqual(len(manifest["variants"]), 5)
        variant = manifest["variants"][0]
        self.assertIn("compile_time_variant", variant)
        self.assertIn("runtime_sweep", variant)
        self.assertIn("barrier_interval", variant["runtime_sweep"]["allowed_fields"])

    def test_parse_benchmark_log_maps_generated_bm_name(self):
        metadata = {
            "rcr_bf16bf16f32_tm8_tn128_tk32_sg1x4_st2_sk1__rcr_bf16_1_4096_14336__screening__0": {
                "shape_id": "rcr_bf16_1_4096_14336",
                "candidate_id": "rcr_bf16bf16f32_tm8_tn128_tk32_sg1x4_st2_sk1",
                "compiler_profile_id": "bmg.small_tile.default",
                "stage": "screening",
                "attempt_index": 0,
                "layout": "rcr",
                "dtype_a": "bf16",
                "dtype_b": "bf16",
                "dtype_c": "f32",
                "dtype_acc": "f32",
                "m": 1,
                "n": 4096,
                "k": 14336,
                "kernel_name": "BmgGemmBF16BF16FP32_RCR_5",
            }
        }
        line = (
            "BmgGemmBF16BF16FP32_RCR_5/"
            "rcr_bf16bf16f32_tm8_tn128_tk32_sg1x4_st2_sk1__rcr_bf16_1_4096_14336__screening__0/"
            "1x4096x14336x1/manual_time avg_runtime_ms=0.412 best_runtime_ms=0.398 "
            "worst_runtime_ms=0.437 avg_tflops=1.13 avg_throughput=287.4\n"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "screening.log"
            log_path.write_text(line, encoding="utf-8")
            rows = profiler.parse_benchmark_log(log_path, metadata, run_id="screening")

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["status"], "pass")
        self.assertEqual(row["verify_status"], "pass")
        self.assertEqual(row["shape_id"], "rcr_bf16_1_4096_14336")
        self.assertEqual(row["avg_tflops"], "1.13")

    def test_dispatch_table_uses_confirmation_median(self):
        shapes = {
            "schema_version": profiler.SCHEMA_VERSION,
            "generated_at": profiler.now_iso(),
            "shape_set_id": "test",
            "source": "predefined",
            "shapes": [
                {
                    "shape_id": "shape_a",
                    "layout": "rcr",
                    "dtype_a": "bf16",
                    "dtype_b": "bf16",
                    "dtype_c": "f32",
                    "dtype_acc": "f32",
                    "m": 1,
                    "n": 4096,
                    "k": 4096,
                }
            ],
        }
        rows = [
            {
                "run_id": "screening",
                "stage": "screening",
                "attempt_index": 0,
                "shape_id": "shape_a",
                "candidate_id": "cand_fast",
                "compiler_profile_id": "bmg.small_tile.default",
                "status": "pass",
                "verify_status": "pass",
                "layout": "rcr",
                "dtype_a": "bf16",
                "dtype_b": "bf16",
                "dtype_c": "f32",
                "dtype_acc": "f32",
                "m": 1,
                "n": 4096,
                "k": 4096,
                "avg_runtime_ms": "0.4",
                "best_runtime_ms": "0.39",
                "worst_runtime_ms": "0.42",
                "avg_tflops": "1.4",
                "avg_throughput": "100",
                "max_error": "",
                "close_call_group": "",
                "failure_reason": "",
                "stdout_log": "a.log",
            },
            {
                "run_id": "confirm",
                "stage": "confirm",
                "attempt_index": 0,
                "shape_id": "shape_a",
                "candidate_id": "cand_fast",
                "compiler_profile_id": "bmg.small_tile.default",
                "status": "pass",
                "verify_status": "pass",
                "layout": "rcr",
                "dtype_a": "bf16",
                "dtype_b": "bf16",
                "dtype_c": "f32",
                "dtype_acc": "f32",
                "m": 1,
                "n": 4096,
                "k": 4096,
                "avg_runtime_ms": "0.5",
                "best_runtime_ms": "0.49",
                "worst_runtime_ms": "0.51",
                "avg_tflops": "1.0",
                "avg_throughput": "90",
                "max_error": "",
                "close_call_group": "",
                "failure_reason": "",
                "stdout_log": "b.log",
            },
            {
                "run_id": "confirm",
                "stage": "confirm",
                "attempt_index": 1,
                "shape_id": "shape_a",
                "candidate_id": "cand_fast",
                "compiler_profile_id": "bmg.small_tile.default",
                "status": "pass",
                "verify_status": "pass",
                "layout": "rcr",
                "dtype_a": "bf16",
                "dtype_b": "bf16",
                "dtype_c": "f32",
                "dtype_acc": "f32",
                "m": 1,
                "n": 4096,
                "k": 4096,
                "avg_runtime_ms": "0.45",
                "best_runtime_ms": "0.44",
                "worst_runtime_ms": "0.46",
                "avg_tflops": "1.1",
                "avg_throughput": "95",
                "max_error": "",
                "close_call_group": "",
                "failure_reason": "",
                "stdout_log": "c.log",
            },
            {
                "run_id": "confirm",
                "stage": "confirm",
                "attempt_index": 0,
                "shape_id": "shape_a",
                "candidate_id": "cand_runner_up",
                "compiler_profile_id": "bmg.small_tile.default",
                "status": "pass",
                "verify_status": "pass",
                "layout": "rcr",
                "dtype_a": "bf16",
                "dtype_b": "bf16",
                "dtype_c": "f32",
                "dtype_acc": "f32",
                "m": 1,
                "n": 4096,
                "k": 4096,
                "avg_runtime_ms": "0.46",
                "best_runtime_ms": "0.45",
                "worst_runtime_ms": "0.47",
                "avg_tflops": "1.05",
                "avg_throughput": "94",
                "max_error": "",
                "close_call_group": "",
                "failure_reason": "",
                "stdout_log": "d.log",
            },
        ]

        dispatch = profiler.build_dispatch_table(rows, shapes, top_k=3, confirm_runs=2, close_call_threshold=10.0)

        self.assertEqual(len(dispatch["entries"]), 1)
        entry = dispatch["entries"][0]
        self.assertEqual(entry["candidate_id"], "cand_fast")
        self.assertTrue(entry["close_call"])
        self.assertAlmostEqual(entry["evidence"]["confirm_median_tflops"], 1.05)

    def test_splitk_candidates_are_gated_to_large_shapes(self):
        shapes = profiler.default_shapes("bf16")
        constraints = profiler.default_constraints()
        profiles = profiler.default_compiler_profiles()
        candidate_space = profiler.generate_candidate_space(
            shapes, constraints, profiles, allowed_runners=("benchmark", "streamk_example")
        )

        small_shape = next(shape for shape in shapes["shapes"] if shape["shape_id"] == "rcr_bf16_8_4096_4096")
        large_shape = next(shape for shape in shapes["shapes"] if shape["shape_id"] == "rcr_bf16_1_4096_14336")

        small_candidates = profiler.choose_candidates_for_shape(small_shape, candidate_space["candidates"])
        large_candidates = profiler.choose_candidates_for_shape(large_shape, candidate_space["candidates"])

        self.assertFalse(any(candidate["split_k"] > 1 for candidate in small_candidates))
        self.assertTrue(any(candidate["split_k"] > 1 for candidate in large_candidates))

    def test_f16_splitk_seed_uses_streamk_runner(self):
        shapes = profiler.default_shapes("f16")
        constraints = profiler.default_constraints()
        profiles = profiler.default_compiler_profiles()
        candidate_space = profiler.generate_candidate_space(
            shapes, constraints, profiles, allowed_runners=("benchmark", "streamk_example")
        )

        splitk = next(candidate for candidate in candidate_space["candidates"] if candidate["split_k"] == 2)
        self.assertEqual(splitk["dtype_a"], "f16")
        self.assertEqual(splitk["runner"], "streamk_example")

    def test_f16_candidate_space_includes_large_tile(self):
        shapes = profiler.default_shapes("f16")
        constraints = profiler.default_constraints()
        profiles = profiler.default_compiler_profiles()
        candidate_space = profiler.generate_candidate_space(shapes, constraints, profiles)

        candidate_ids = {candidate["candidate_id"] for candidate in candidate_space["candidates"]}
        self.assertIn("rcr_f16f16f32_tm256_tn256_tk32_sg8x4_st2_sk1", candidate_ids)

    def test_dry_run_shapes_use_tiny_shape_set(self):
        shapes = profiler.dry_run_shapes("bf16")

        self.assertEqual(shapes["source"], "dry_run")
        self.assertEqual(len(shapes["shapes"]), 1)
        shape = shapes["shapes"][0]
        self.assertEqual((shape["m"], shape["n"], shape["k"]), (1, 64, 32))
        self.assertEqual(shape["runtime_defaults"]["barrier_interval"], 0)

    def test_dry_run_workflow_uses_minimal_shape_set_and_no_confirmation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = profiler.build_parser().parse_args(
                [
                    "--workspace",
                    tmpdir,
                    "--dtype",
                    "f16",
                    "--dry-run",
                    "--skip-run",
                ]
            )
            outputs = profiler.workflow(args)

            self.assertTrue(outputs["dry_run"])
            shapes_doc = profiler.read_json(Path(tmpdir) / "inputs" / "gemm_target_shapes.json")
            self.assertEqual(shapes_doc["source"], "dry_run")
            phase_a_summary = profiler.read_json(Path(tmpdir) / "reports" / "phase_a_summary.json")
            self.assertEqual(phase_a_summary["probe_mode"], "dry_run_off")
            phase_b_summary = profiler.read_json(Path(tmpdir) / "reports" / "phase_b_summary.json")
            self.assertEqual(phase_b_summary["candidate_count"], 5)

    def test_static_probe_disables_splitk_without_streamk_binary(self):
        constraints = profiler.default_constraints()
        env_caps = {
            "executables": {
                "benchmark_available": True,
                "streamk_example_available": False,
            }
        }
        probed = profiler.apply_static_probe_constraints(constraints, env_caps)
        self.assertEqual(probed["limits"]["max_split_k"], 1)
        self.assertEqual(probed["allowed_values"]["split_k"], [1])

    def test_run_probe_adds_blocked_rule_for_failed_candidate(self):
        constraints = profiler.default_constraints()
        rows = [
            {
                "candidate_id": "rcr_bf16bf16f32_tm8_tn64_tk32_sg1x4_st2_sk2",
                "status": "fail",
                "split_k": "2",
            }
        ]
        probed = profiler.apply_run_probe_constraints(constraints, rows)
        self.assertEqual(probed["limits"]["max_split_k"], 1)
        self.assertEqual(probed["allowed_values"]["split_k"], [1])
        self.assertEqual(len(probed["blocked_rules"]), 1)
        self.assertEqual(probed["blocked_rules"][0]["match"]["tile_m"], 8)


if __name__ == "__main__":
    unittest.main()
