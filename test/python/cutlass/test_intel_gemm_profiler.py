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
        self.assertEqual(len(candidate_space["candidates"]), 8)
        candidate_ids = {candidate["candidate_id"] for candidate in candidate_space["candidates"]}
        self.assertIn("rcr_bf16bf16f32_tm8_tn128_tk32_sg1x4_st2_sk1", candidate_ids)
        self.assertIn("rcr_bf16bf16f32_tm256_tn256_tk32_sg8x4_st2_sk1", candidate_ids)
        self.assertIn("rcr_bf16bf16f32_tm128_tn256_tk32_sg4x4_st2_sk1", candidate_ids)
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
        self.assertEqual(len(catalog["kernels"]), 9)
        splitk = next(entry for entry in catalog["kernels"] if entry["split_k"] == 2)
        self.assertEqual(splitk["instantiation_level"], 0)
        self.assertEqual(splitk["runtime_defaults"], {})
        self.assertEqual(splitk["allowed_runtime_sweeps"], ["shape_id", "m", "n", "k"])

    def test_kernel_catalog_prefers_repo_json(self):
        catalog = profiler.load_persisted_kernel_catalog()

        self.assertEqual(catalog["catalog_version"], "level0-seed-catalog")
        kernel_ids = {entry["kernel_id"] for entry in catalog["kernels"]}
        self.assertIn("BmgGemmFP16FP16FP32_RCR_6", kernel_ids)

    def test_generated_seed_catalog_matches_persisted_catalog(self):
        generated = profiler.generated_level0_kernel_catalog()
        persisted = profiler.load_persisted_kernel_catalog()

        normalize = lambda catalog: sorted(catalog["kernels"], key=lambda entry: entry["kernel_id"])
        self.assertEqual(generated["catalog_version"], persisted["catalog_version"])
        self.assertEqual(generated["search_runtime_schema"], persisted["search_runtime_schema"])
        self.assertEqual(normalize(generated), normalize(persisted))

    def test_build_candidate_build_manifest_splits_compile_and_runtime_fields(self):
        shapes = profiler.default_shapes("bf16")
        constraints = profiler.default_constraints()
        profiles = profiler.default_compiler_profiles()
        candidate_space = profiler.generate_candidate_space(shapes, constraints, profiles)

        manifest = profiler.build_candidate_build_manifest(candidate_space)

        self.assertEqual(manifest["search_runtime_schema"]["microbench_guided_defaults"]["grf_mode"], 256)
        self.assertEqual(len(manifest["variants"]), 8)
        variant = manifest["variants"][0]
        self.assertIn("compile_time_variant", variant)
        self.assertIn("runtime_sweep", variant)
        self.assertEqual(variant["runtime_sweep"]["allowed_fields"], ["shape_id", "m", "n", "k"])

    def test_default_constraints_use_calibrated_slm_limit(self):
        constraints = profiler.default_constraints()

        self.assertEqual(constraints["limits"]["max_slm_kb"], 64)

    def test_default_compiler_profiles_split_build_and_runtime_config(self):
        profiles = profiler.default_compiler_profiles()

        self.assertEqual(
            profiles["build_config"]["selected_compile_variant"],
            "perf_default",
        )
        self.assertEqual(
            profiles["build_config"]["cmake_vars"]["CUTLASS_SYCL_PROFILING_ENABLED"],
            "OFF",
        )
        self.assertEqual(
            profiles["build_config"]["cmake_vars"]["CUTLASS_ENABLE_EXAMPLES"],
            "OFF",
        )
        self.assertEqual(
            profiles["build_config"]["cmake_vars"]["CUTLASS_ENABLE_TESTS"],
            "OFF",
        )
        self.assertNotIn("IGC_VISAOptions", profiles["build_config"]["compile_env"])
        self.assertEqual(
            profiles["runtime_config"]["runtime_env"]["ONEAPI_DEVICE_SELECTOR"],
            "level_zero:gpu",
        )
        self.assertEqual(profiles["profiles"][0]["runtime_env_override"], {})
        self.assertEqual(
            profiles["build_config"]["compile_env_variant_metadata"]["perf_default"]["status"],
            "validated",
        )

    def test_build_config_marks_128grf_as_experimental(self):
        profiles = profiler.default_compiler_profiles()

        experiment = profiles["build_config"]["compile_env_variants"]["perf_128grf_experiment"]
        metadata = profiles["build_config"]["compile_env_variant_metadata"]["perf_128grf_experiment"]

        self.assertEqual(experiment["IGC_TotalGRFNum"], "128")
        self.assertNotIn("SYCL_PROGRAM_COMPILE_OPTIONS", experiment)
        self.assertEqual(metadata["status"], "needs_validation")
        self.assertIn("Advisory only", metadata["notes"])
        self.assertIn("do not pass -cl-intel-256-GRF-per-thread", metadata["notes"])
        self.assertIn("do not use in production", metadata["notes"])

    def test_load_persisted_build_config_fallback_matches_experimental_variants(self):
        persisted = profiler.load_persisted_build_config(profiler.DEFAULT_BUILD_CONFIG_PATH)
        with tempfile.TemporaryDirectory() as tmpdir:
            fallback_path = Path(tmpdir) / "missing_build_config.json"
            build_config = profiler.load_persisted_build_config(fallback_path)

        variants = build_config["compile_env_variants"]

        self.assertIn("perf_default", variants)
        self.assertIn("perf_perfmodel", variants)
        self.assertIn("perf_128grf_experiment", variants)
        self.assertIn("perf_enableBCR", variants)
        self.assertIn("debug_with_lines", variants)
        self.assertEqual(
            variants["perf_perfmodel"]["IGC_VISAOptions"],
            "-perfmodel",
        )
        self.assertEqual(
            variants["perf_enableBCR"]["IGC_VISAOptions"],
            "-enableBCR",
        )
        self.assertEqual(
            variants["debug_with_lines"]["SYCL_PROGRAM_COMPILE_OPTIONS"],
            "-ze-opt-large-register-file -gline-tables-only",
        )
        self.assertEqual(
            variants["perf_default"]["SYCL_PROGRAM_COMPILE_OPTIONS"],
            "-ze-opt-large-register-file",
        )
        self.assertEqual(
            build_config["compile_env_variant_metadata"]["perf_128grf_experiment"]["status"],
            "needs_validation",
        )
        self.assertEqual(build_config["compile_env_variants"], persisted["compile_env_variants"])
        self.assertEqual(
            build_config["compile_env_variant_metadata"],
            persisted["compile_env_variant_metadata"],
        )

    def test_selected_runtime_env_ignores_compile_time_flags(self):
        profiles = profiler.default_compiler_profiles()

        runtime_env = profiler.selected_runtime_env(profiles, profiles["profiles"][0])

        self.assertEqual(runtime_env["ONEAPI_DEVICE_SELECTOR"], "level_zero:gpu")
        self.assertNotIn("IGC_ExtraOCLOptions", runtime_env)
        self.assertNotIn("IGC_VISAOptions", runtime_env)
        self.assertNotIn("SYCL_PROGRAM_COMPILE_OPTIONS", runtime_env)

    def test_resolve_hw_reference_spec_uses_calibrated_b60_data(self):
        spec = profiler.resolve_hw_reference_spec("bmg")

        self.assertEqual(spec["device_id"], "bmg_g21")
        self.assertEqual(spec["clock_mhz"], 2400)
        self.assertEqual(spec["peak_bf16_tflops"], 97.66)
        self.assertEqual(spec["measured_read_bw_gbps"], 538)
        self.assertEqual(spec["slm_per_xe_core_kb"], 64)
        self.assertEqual(spec["calibration_status"], "measured")
        self.assertEqual(spec["concurrent_sgs_per_xe_core_128grf"], 16)
        self.assertEqual(spec["grf_bytes_per_thread_128grf"], 4096)

    def test_resolve_hw_reference_spec_marks_b70_as_not_measured(self):
        spec = profiler.resolve_hw_reference_spec(hw_spec_id="bmg_g31")

        self.assertEqual(spec["device_id"], "bmg_g31")
        self.assertEqual(spec["calibration_status"], "not_measured")

    def test_select_compiler_profile_skips_failed_probe_profile(self):
        profiles = profiler.default_compiler_profiles()
        profiles["profiles"][0]["probe_status"] = "fail"

        selected = profiler.select_compiler_profile_id(profiles, tile_m=8, sg_count=4)

        self.assertNotEqual(selected, "bmg.small_tile.default")

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

    def test_parse_benchmark_log_does_not_treat_max_error_metric_as_failure(self):
        metadata = {
            "bm_case": {
                "shape_id": "shape_a",
                "candidate_id": "cand_a",
                "compiler_profile_id": "bmg.small_tile.default",
                "stage": "screening",
                "attempt_index": 0,
                "layout": "rcr",
                "dtype_a": "bf16",
                "dtype_b": "bf16",
                "dtype_c": "f32",
                "dtype_acc": "f32",
                "m": 1,
                "n": 64,
                "k": 32,
            }
        }
        line = (
            "Kernel/bm_case/1x64x32x1/manual_time avg_runtime_ms=0.1 best_runtime_ms=0.09 "
            "worst_runtime_ms=0.11 avg_tflops=1.2 avg_throughput=3.4 max_error=0.0001\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "parser.log"
            log_path.write_text(line, encoding="utf-8")
            rows = profiler.parse_benchmark_log(log_path, metadata, run_id="screening")

        self.assertEqual(rows[0]["status"], "pass")
        self.assertEqual(rows[0]["failure_reason"], "")

    def test_parse_benchmark_log_detects_real_error_line(self):
        metadata = {
            "bm_case": {
                "shape_id": "shape_a",
                "candidate_id": "cand_a",
                "compiler_profile_id": "bmg.small_tile.default",
                "stage": "screening",
                "attempt_index": 0,
                "layout": "rcr",
                "dtype_a": "bf16",
                "dtype_b": "bf16",
                "dtype_c": "f32",
                "dtype_acc": "f32",
                "m": 1,
                "n": 64,
                "k": 32,
            }
        }
        line = "Kernel/bm_case/1x64x32x1 ERROR OCCURRED can_implement failed\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "parser_error.log"
            log_path.write_text(line, encoding="utf-8")
            rows = profiler.parse_benchmark_log(log_path, metadata, run_id="screening")

        self.assertEqual(rows[0]["status"], "fail")
        self.assertIn("ERROR OCCURRED", rows[0]["failure_reason"])

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

    def test_choose_candidates_handles_m16_and_m128_boundaries(self):
        constraints = profiler.default_constraints()
        profiles = profiler.default_compiler_profiles()
        shapes = {
            "schema_version": profiler.SCHEMA_VERSION,
            "generated_at": profiler.now_iso(),
            "shape_set_id": "boundary",
            "source": "test",
            "shapes": [
                {
                    "shape_id": "shape_m16",
                    "layout": "rcr",
                    "dtype_a": "bf16",
                    "dtype_b": "bf16",
                    "dtype_c": "f32",
                    "dtype_acc": "f32",
                    "m": 16,
                    "n": 4096,
                    "k": 4096,
                },
                {
                    "shape_id": "shape_m128",
                    "layout": "rcr",
                    "dtype_a": "bf16",
                    "dtype_b": "bf16",
                    "dtype_c": "f32",
                    "dtype_acc": "f32",
                    "m": 128,
                    "n": 4096,
                    "k": 8192,
                },
            ],
        }
        candidate_space = profiler.generate_candidate_space(shapes, constraints, profiles)

        m16_candidates = profiler.choose_candidates_for_shape(shapes["shapes"][0], candidate_space["candidates"])
        m128_candidates = profiler.choose_candidates_for_shape(shapes["shapes"][1], candidate_space["candidates"])

        self.assertTrue(any(candidate["tile_m"] == 128 for candidate in m16_candidates))
        self.assertTrue(any(candidate["tile_m"] == 256 for candidate in m128_candidates))

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
        self.assertIn("rcr_f16f16f32_tm128_tn256_tk32_sg4x4_st2_sk1", candidate_ids)

    def test_dry_run_shapes_use_tiny_shape_set(self):
        shapes = profiler.dry_run_shapes("bf16")

        self.assertEqual(shapes["source"], "dry_run")
        self.assertEqual(len(shapes["shapes"]), 1)
        shape = shapes["shapes"][0]
        self.assertEqual((shape["m"], shape["n"], shape["k"]), (1, 64, 32))
        self.assertEqual(shape["runtime_defaults"], {})

    def test_build_dpas_probe_entry_uses_small_benchmark_candidate(self):
        shapes = profiler.default_shapes("bf16")
        constraints = profiler.default_constraints()
        profiles = profiler.default_compiler_profiles()
        candidate_space = profiler.generate_candidate_space(shapes, constraints, profiles, allowed_runners=("benchmark",))

        entry = profiler.build_dpas_probe_entry(shapes, candidate_space)

        self.assertIsNotNone(entry)
        self.assertEqual(entry["stage"], "dpas_probe")
        self.assertEqual(entry["candidate"]["candidate_id"], "rcr_bf16bf16f32_tm8_tn64_tk32_sg1x4_st2_sk1")
        self.assertEqual(entry["shape"]["shape_id"], "rcr_bf16_8_4096_4096")

    def test_build_phase_a_probe_entries_use_dynamic_shapes(self):
        shapes = {
            "schema_version": profiler.SCHEMA_VERSION,
            "generated_at": profiler.now_iso(),
            "shape_set_id": "custom",
            "source": "custom",
            "shapes": [
                {
                    "shape_id": "custom_small",
                    "layout": "rcr",
                    "dtype_a": "bf16",
                    "dtype_b": "bf16",
                    "dtype_c": "f32",
                    "dtype_acc": "f32",
                    "m": 4,
                    "n": 2048,
                    "k": 2048,
                },
                {
                    "shape_id": "custom_mid",
                    "layout": "rcr",
                    "dtype_a": "bf16",
                    "dtype_b": "bf16",
                    "dtype_c": "f32",
                    "dtype_acc": "f32",
                    "m": 48,
                    "n": 4096,
                    "k": 4096,
                },
                {
                    "shape_id": "custom_big",
                    "layout": "rcr",
                    "dtype_a": "bf16",
                    "dtype_b": "bf16",
                    "dtype_c": "f32",
                    "dtype_acc": "f32",
                    "m": 192,
                    "n": 4096,
                    "k": 8192,
                },
            ],
        }
        constraints = profiler.default_constraints()
        profiles = profiler.default_compiler_profiles()
        candidate_space = profiler.generate_candidate_space(shapes, constraints, profiles, allowed_runners=("benchmark",))

        entries = profiler.build_phase_a_probe_entries(shapes, candidate_space)
        shape_ids = {entry["shape"]["shape_id"] for entry in entries}

        self.assertIn("custom_small", shape_ids)
        self.assertIn("custom_mid", shape_ids)
        self.assertIn("custom_big", shape_ids)

    def test_build_compiler_profile_probe_entries_matches_candidate_classes(self):
        shapes = profiler.default_shapes("bf16")
        constraints = profiler.default_constraints()
        profiles = profiler.default_compiler_profiles()
        candidate_space = profiler.generate_candidate_space(shapes, constraints, profiles, allowed_runners=("benchmark",))

        entries = profiler.build_compiler_profile_probe_entries(shapes, candidate_space, profiles)

        self.assertEqual(len(entries), 3)
        probe_ids = {entry["compiler_profile_probe_id"] for entry in entries}
        self.assertIn("bmg.small_tile.default", probe_ids)
        self.assertIn("bmg.medium_tile.default", probe_ids)
        self.assertIn("bmg.large_tile.default", probe_ids)
        self.assertEqual(entries[0]["compiler_profile_id"], entries[0]["compiler_profile_probe_id"])

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
            self.assertEqual(phase_b_summary["candidate_count"], 8)

    def test_dispatch_table_reports_low_efficiency_winner(self):
        shapes = {
            "schema_version": profiler.SCHEMA_VERSION,
            "generated_at": profiler.now_iso(),
            "shape_set_id": "efficiency",
            "source": "test",
            "shapes": [
                {
                    "shape_id": "shape_peak_gap",
                    "layout": "rcr",
                    "dtype_a": "bf16",
                    "dtype_b": "bf16",
                    "dtype_c": "f32",
                    "dtype_acc": "f32",
                    "m": 256,
                    "n": 4096,
                    "k": 8192,
                }
            ],
        }
        constraints = profiler.default_constraints()
        profiles = profiler.default_compiler_profiles()
        candidate_space = profiler.generate_candidate_space(shapes, constraints, profiles)
        rows = [
            {
                "run_id": "confirm",
                "stage": "confirm",
                "attempt_index": 0,
                "shape_id": "shape_peak_gap",
                "candidate_id": "rcr_bf16bf16f32_tm16_tn64_tk32_sg2x4_st2_sk1",
                "compiler_profile_id": "bmg.small_tile.default",
                "status": "pass",
                "verify_status": "pass",
                "layout": "rcr",
                "dtype_a": "bf16",
                "dtype_b": "bf16",
                "dtype_c": "f32",
                "dtype_acc": "f32",
                "m": 256,
                "n": 4096,
                "k": 8192,
                "avg_runtime_ms": "0.57",
                "best_runtime_ms": "0.56",
                "worst_runtime_ms": "0.58",
                "avg_tflops": "10.0",
                "avg_throughput": "0",
                "max_error": "",
                "close_call_group": "",
                "failure_reason": "",
                "stdout_log": "winner.log",
            }
        ]

        dispatch = profiler.build_dispatch_table(
            rows,
            shapes,
            top_k=3,
            confirm_runs=1,
            close_call_threshold=3.0,
            candidate_space=candidate_space,
            hw_spec=profiler.resolve_hw_reference_spec("bmg"),
        )

        entry = dispatch["entries"][0]
        self.assertEqual(entry["candidate_id"], "rcr_bf16bf16f32_tm16_tn64_tk32_sg2x4_st2_sk1")
        self.assertEqual(entry["efficiency_warning"], "winner_efficiency_below_40pct_peak")
        self.assertLess(entry["selected_efficiency"], 0.4)

    def test_phase_b_summary_surfaces_low_efficiency_warnings(self):
        candidate_space = {"candidates": [{}], "kernel_catalog": {"catalog_version": "test-catalog"}}
        dispatch_table = {
            "entries": [
                {
                    "shape_id": "shape_peak_gap",
                    "candidate_id": "cand_a",
                    "selected_efficiency": 0.31,
                    "peak_tflops": 97.66,
                    "efficiency_warning": "winner_efficiency_below_40pct_peak",
                }
            ]
        }
        summary = {"rows": 10, "passed": 10, "failed": 0}

        phase_b_summary = profiler.build_phase_b_summary(candidate_space, dispatch_table, summary)

        self.assertEqual(len(phase_b_summary["low_efficiency_warnings"]), 1)
        self.assertEqual(
            phase_b_summary["low_efficiency_warnings"][0]["warning"],
            "winner_efficiency_below_40pct_peak",
        )

    def test_dispatch_table_skips_low_efficiency_warning_for_memory_bound_winner(self):
        shapes = {
            "schema_version": profiler.SCHEMA_VERSION,
            "generated_at": profiler.now_iso(),
            "shape_set_id": "memory-bound",
            "source": "test",
            "shapes": [
                {
                    "shape_id": "decode_shape",
                    "layout": "rcr",
                    "dtype_a": "bf16",
                    "dtype_b": "bf16",
                    "dtype_c": "f32",
                    "dtype_acc": "f32",
                    "m": 1,
                    "n": 4096,
                    "k": 14336,
                }
            ],
        }
        constraints = profiler.default_constraints()
        profiles = profiler.default_compiler_profiles()
        candidate_space = profiler.generate_candidate_space(shapes, constraints, profiles)
        rows = [
            {
                "run_id": "confirm",
                "stage": "confirm",
                "attempt_index": 0,
                "shape_id": "decode_shape",
                "candidate_id": "rcr_bf16bf16f32_tm8_tn128_tk32_sg1x8_st2_sk1",
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
                "k": 14336,
                "avg_runtime_ms": "0.235",
                "best_runtime_ms": "0.230",
                "worst_runtime_ms": "0.240",
                "avg_tflops": "0.5",
                "avg_throughput": "0",
                "max_error": "",
                "close_call_group": "",
                "failure_reason": "",
                "stdout_log": "decode.log",
            }
        ]

        dispatch = profiler.build_dispatch_table(
            rows,
            shapes,
            top_k=1,
            confirm_runs=1,
            close_call_threshold=3.0,
            candidate_space=candidate_space,
            hw_spec=profiler.resolve_hw_reference_spec("bmg"),
        )

        entry = dispatch["entries"][0]
        self.assertLess(entry["selected_efficiency"], 0.4)
        self.assertEqual(entry["efficiency_warning"], "")

    def test_generate_candidate_space_rejects_unsupported_layout(self):
        shapes = {
            "schema_version": profiler.SCHEMA_VERSION,
            "generated_at": profiler.now_iso(),
            "shape_set_id": "unsupported-layout",
            "source": "test",
            "shapes": [
                {
                    "shape_id": "shape_rrr",
                    "layout": "rrr",
                    "dtype_a": "bf16",
                    "dtype_b": "bf16",
                    "dtype_c": "f32",
                    "dtype_acc": "f32",
                    "m": 64,
                    "n": 256,
                    "k": 256,
                }
            ],
        }

        with self.assertRaisesRegex(ValueError, "Unsupported layouts in shapes: rrr"):
            profiler.generate_candidate_space(shapes, profiler.default_constraints(), profiler.default_compiler_profiles())

    def test_phase_a_summary_includes_dpas_probe(self):
        summary = profiler.build_phase_a_summary(
            {
                "probe_mode": "run",
                "dpas_baseline_probe": {"status": "pass", "avg_tflops": "1.23"},
                "compiler_flags_probe": {"results": [{"compiler_profile_id": "bmg.small_tile.default", "status": "pass"}]},
            },
            profiler.default_constraints(),
            [],
        )

        self.assertEqual(summary["dpas_baseline_probe"]["status"], "pass")
        self.assertEqual(summary["compiler_flags_probe"]["results"][0]["compiler_profile_id"], "bmg.small_tile.default")

    def test_apply_probe_results_to_profiles_marks_selected_profiles(self):
        profiles = profiler.default_compiler_profiles()
        summary = {
            "results": [
                {"compiler_profile_id": "bmg.small_tile.default", "status": "pass", "avg_tflops": "1.0", "avg_runtime_ms": "0.1"},
                {"compiler_profile_id": "bmg.medium_tile.default", "status": "fail", "avg_tflops": "", "avg_runtime_ms": ""},
                {"compiler_profile_id": "bmg.large_tile.default", "status": "pass", "avg_tflops": "2.0", "avg_runtime_ms": "0.2"},
            ],
            "selected_profile_ids": {"small_tile": "bmg.small_tile.default", "large_tile": "bmg.large_tile.default"},
        }

        updated = profiler.apply_probe_results_to_profiles(profiles, summary)

        by_id = {profile["compiler_profile_id"]: profile for profile in updated["profiles"]}
        self.assertEqual(by_id["bmg.small_tile.default"]["probe_status"], "pass")
        self.assertTrue(by_id["bmg.small_tile.default"]["probe_selected"])
        self.assertEqual(by_id["bmg.medium_tile.default"]["probe_status"], "fail")
        self.assertFalse(by_id["bmg.medium_tile.default"]["probe_selected"])

    def test_build_compiler_flags_probe_summary_groups_by_candidate_class(self):
        profiles = profiler.default_compiler_profiles()
        rows = [
            {
                "compiler_profile_id": "bmg.small_tile.default",
                "candidate_id": "cand_small_a",
                "shape_id": "shape_small_a",
                "status": "pass",
                "avg_tflops": "1.0",
                "avg_runtime_ms": "0.10",
                "stdout_log": "small_a.log",
            },
            {
                "compiler_profile_id": "bmg.small_tile.default",
                "candidate_id": "cand_small_b",
                "shape_id": "shape_small_b",
                "status": "pass",
                "avg_tflops": "1.4",
                "avg_runtime_ms": "0.08",
                "stdout_log": "small_b.log",
            },
            {
                "compiler_profile_id": "bmg.large_tile.default",
                "candidate_id": "cand_large",
                "shape_id": "shape_large",
                "status": "fail",
                "avg_tflops": "",
                "avg_runtime_ms": "",
                "stdout_log": "large.log",
            },
        ]

        summary = profiler.build_compiler_flags_probe_summary(rows, profiles)

        self.assertEqual(summary["selected_profile_ids"]["small_tile"], "bmg.small_tile.default")
        results_by_id = {item["compiler_profile_id"]: item for item in summary["results"]}
        self.assertEqual(results_by_id["bmg.small_tile.default"]["candidate_class"], "small_tile")
        self.assertEqual(results_by_id["bmg.small_tile.default"]["samples"], 2)
        self.assertEqual(results_by_id["bmg.small_tile.default"]["avg_tflops"], "1.2")
        self.assertEqual(results_by_id["bmg.large_tile.default"]["candidate_class"], "large_tile")

    def test_build_compiler_flags_probe_summary_handles_empty_pass_metrics(self):
        profiles = profiler.default_compiler_profiles()
        rows = [
            {
                "compiler_profile_id": "bmg.small_tile.default",
                "candidate_id": "cand_small",
                "shape_id": "shape_small",
                "status": "pass",
                "avg_tflops": "",
                "avg_runtime_ms": "",
                "stdout_log": "small.log",
            }
        ]

        summary = profiler.build_compiler_flags_probe_summary(rows, profiles)

        result = summary["results"][0]
        self.assertEqual(result["avg_tflops"], "")
        self.assertEqual(result["avg_runtime_ms"], "")
        self.assertEqual(summary["selected_profile_ids"]["small_tile"], "bmg.small_tile.default")

    def test_runner_environment_metadata_uses_shared_schema_version(self):
        metadata = profiler.collect_environment_metadata("", "missing-benchmark", "missing-streamk")

        self.assertEqual(metadata["schema_version"], profiler.SCHEMA_VERSION)
        self.assertRegex(metadata["python_version"], r"^\d+\.\d+\.\d+")

    def test_run_benchmark_returns_timeout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "timeout.log"
            process, timed_out, reason = profiler.run_benchmark(
                ["python3", "-c", "import time; time.sleep(2)"],
                log_path,
                timeout=1,
            )

        self.assertTrue(timed_out)
        self.assertEqual(process.returncode, 124)
        self.assertIn("timeout after 1s", reason)

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

    def test_detect_probe_anomalies_blocks_large_tile_regression(self):
        shapes = profiler.default_shapes("bf16")
        constraints = profiler.default_constraints()
        profiles = profiler.default_compiler_profiles()
        candidate_space = profiler.generate_candidate_space(shapes, constraints, profiles, allowed_runners=("benchmark",))
        hw_spec = profiler.resolve_hw_reference_spec("bmg")
        small_candidate = min(
            (candidate for candidate in candidate_space["candidates"] if candidate["candidate_class"] == "small_tile"),
            key=lambda candidate: (candidate["tile_m"], candidate["tile_n"]),
        )
        large_candidate = max(
            (candidate for candidate in candidate_space["candidates"] if candidate["candidate_class"] == "large_tile"),
            key=lambda candidate: (candidate["tile_m"], candidate["tile_n"]),
        )
        probe_rows = [
            {
                "candidate_id": small_candidate["candidate_id"],
                "shape_id": "rcr_bf16_8_4096_4096",
                "status": "pass",
                "avg_tflops": "2.997",
            },
            {
                "candidate_id": small_candidate["candidate_id"],
                "shape_id": "rcr_bf16_64_4096_4096",
                "status": "pass",
                "avg_tflops": "18.9",
            },
            {
                "candidate_id": large_candidate["candidate_id"],
                "shape_id": "rcr_bf16_256_4096_8192",
                "status": "pass",
                "avg_tflops": "2.282",
            },
        ]

        report = profiler.detect_probe_anomalies(probe_rows, shapes, candidate_space, hw_spec)

        self.assertEqual(report["hw_spec"], "bmg_g21")
        self.assertEqual(len(report["anomalies"]), 1)
        anomaly = report["anomalies"][0]
        self.assertEqual(anomaly["candidate_id"], large_candidate["candidate_id"])
        self.assertEqual(anomaly["spec_anomaly"], "severely_below_spec")
        self.assertTrue(anomaly["cross_anomaly"].startswith("large_tile_slower_than_"))
        self.assertEqual(report["auto_block_rules"][0]["rule_id"], f"probe.auto_block.anomaly.{large_candidate['candidate_id']}")

    def test_apply_run_probe_constraints_appends_anomaly_auto_block(self):
        constraints = profiler.default_constraints()
        rows = []
        anomaly_report = {
            "auto_block_rules": [
                {
                    "rule_id": "probe.auto_block.anomaly.test_candidate",
                    "match": {"tile_m": 256, "tile_n": 256, "tile_k": 32, "sg_m": 8, "sg_n": 4, "split_k": 1},
                    "reason": "severely_below_spec",
                    "evidence_tflops": 2.282,
                }
            ]
        }

        probed = profiler.apply_run_probe_constraints(constraints, rows, anomaly_report=anomaly_report)

        self.assertEqual(probed["limits"]["max_slm_kb"], 64)
        self.assertEqual(len(probed["blocked_rules"]), 1)
        self.assertEqual(probed["blocked_rules"][0]["rule_id"], "probe.auto_block.anomaly.test_candidate")

    def test_compute_efficiency_bounds_handles_memory_bound_decode_shape(self):
        hw_spec = profiler.resolve_hw_reference_spec("bmg")
        shape = profiler.default_shapes("bf16")["shapes"][0]
        candidate = {
            "tile_m": 8,
            "tile_n": 64,
            "tile_k": 32,
            "sg_m": 1,
            "sg_n": 4,
            "stages": 2,
            "dtype_a": "bf16",
        }

        min_eff, max_eff = profiler.compute_efficiency_bounds(shape, candidate, hw_spec)

        self.assertLess(min_eff, max_eff)
        self.assertLess(max_eff, 0.02)

    def test_detect_probe_anomalies_above_expected_is_report_only(self):
        shapes = profiler.default_shapes("bf16")
        constraints = profiler.default_constraints()
        profiles = profiler.default_compiler_profiles()
        candidate_space = profiler.generate_candidate_space(shapes, constraints, profiles, allowed_runners=("benchmark",))
        hw_spec = profiler.resolve_hw_reference_spec("bmg")
        small_candidate = min(
            (candidate for candidate in candidate_space["candidates"] if candidate["candidate_class"] == "small_tile"),
            key=lambda candidate: (candidate["tile_m"], candidate["tile_n"]),
        )
        probe_rows = [
            {
                "candidate_id": small_candidate["candidate_id"],
                "shape_id": "rcr_bf16_8_4096_4096",
                "status": "pass",
                "avg_tflops": "20.0",
            }
        ]

        report = profiler.detect_probe_anomalies(probe_rows, shapes, candidate_space, hw_spec)

        self.assertEqual(report["hw_spec_calibration_status"], "measured")
        self.assertEqual(report["anomalies"][0]["spec_anomaly"], "above_expected")
        self.assertEqual(report["anomalies"][0]["auto_action"], "reported")
        self.assertEqual(report["auto_block_rules"], [])

    def test_detect_probe_anomalies_uses_fastest_cross_reference(self):
        shapes = profiler.default_shapes("bf16")
        constraints = profiler.default_constraints()
        profiles = profiler.default_compiler_profiles()
        candidate_space = profiler.generate_candidate_space(shapes, constraints, profiles, allowed_runners=("benchmark",))
        hw_spec = profiler.resolve_hw_reference_spec("bmg")
        small_candidates = sorted(
            (candidate for candidate in candidate_space["candidates"] if candidate["candidate_class"] == "small_tile"),
            key=lambda candidate: (candidate["tile_m"], candidate["tile_n"]),
        )
        large_candidate = max(
            (candidate for candidate in candidate_space["candidates"] if candidate["candidate_class"] == "large_tile"),
            key=lambda candidate: (candidate["tile_m"], candidate["tile_n"]),
        )
        probe_rows = [
            {
                "candidate_id": small_candidates[0]["candidate_id"],
                "shape_id": "rcr_bf16_8_4096_4096",
                "status": "pass",
                "avg_tflops": "10.0",
            },
            {
                "candidate_id": small_candidates[1]["candidate_id"],
                "shape_id": "rcr_bf16_64_4096_4096",
                "status": "pass",
                "avg_tflops": "20.0",
            },
            {
                "candidate_id": large_candidate["candidate_id"],
                "shape_id": "rcr_bf16_256_4096_8192",
                "status": "pass",
                "avg_tflops": "2.0",
            },
        ]

        report = profiler.detect_probe_anomalies(probe_rows, shapes, candidate_space, hw_spec)

        large_anomaly = next(
            item for item in report["anomalies"] if item["candidate_id"] == large_candidate["candidate_id"]
        )
        self.assertEqual(
            large_anomaly["cross_anomaly"],
            f"large_tile_slower_than_{small_candidates[1]['candidate_id']}",
        )

    def test_phase_a_summary_includes_anomaly_report(self):
        summary = profiler.build_phase_a_summary(
            {
                "probe_mode": "run",
                "hw_reference_spec_id": "bmg_g21",
                "dpas_baseline_probe": {"status": "pass", "avg_tflops": "1.23"},
                "compiler_flags_probe": {"results": [{"compiler_profile_id": "bmg.small_tile.default", "status": "pass"}]},
                "anomaly_report": {"anomalies": [{"candidate_id": "cand_large"}], "auto_block_rules": [{"rule_id": "rule.large"}]},
            },
            profiler.default_constraints(),
            [],
        )

        self.assertEqual(summary["hw_reference_spec_id"], "bmg_g21")
        self.assertEqual(summary["anomaly_report"]["anomalies"][0]["candidate_id"], "cand_large")


if __name__ == "__main__":
    unittest.main()
