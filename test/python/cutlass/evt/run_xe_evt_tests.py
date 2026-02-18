#################################################################################################
# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################################

import argparse
import sys
import unittest

from utils.test_report import write_test_results_to_csv, print_test_summary, TestResultWithSuccesses


# Define test suites - each suite contains a list of test modules
TEST_SUITES = {
    'xe_evt_ci': [
        'evt_compute_xe_sm80_90.TestEVTCompute.test_arith',
        'evt_compute_xe_sm80_90.TestEVTCompute.test_func_call',
        'evt_compute_xe_sm80_90.TestEVTCompute.test_func_call2',
        'evt_compute_xe_sm80_90.TestEVTCompute.test_gelu',
        'evt_compute_xe_sm80_90.TestEVTCompute.test_sigmoid',
        'evt_compute_xe_sm80_90.TestEVTCompute.test_tanh',
        'evt_compute_xe_sm80_90.TestEVTCompute.test_exp',
        'evt_load_xe_sm80_90.TestEVTLoad.test_tensor_load',
        'evt_load_xe_sm80_90.TestEVTLoad.test_row_broadcast',
        'evt_load_xe_sm80_90.TestEVTLoad.test_column_broadcast',
        'evt_load_xe_sm80_90.TestEVTLoad.test_scalar_broadcast',
        'evt_store_xe_sm80_90.TestEVTStore.test_invalid_store',
        'evt_store_xe_sm80_90.TestEVTStore.test_aux_store',
        'evt_store_xe_sm80_90.TestEVTStore.test_col_reduce',
        'evt_store_xe_sm80_90.TestEVTStore.test_row_reduce',
        'evt_store_xe_sm80_90.TestEVTStore.test_scalar_reduce',
        'evt_store_xe_sm80_90.TestEVTStore.test_store_with_multiple_reductions',
        'evt_mixed_xe_sm80_90.TestEVTMixed.test_same_variable_used_multiple_times',
        'evt_mixed_xe_sm80_90.TestEVTMixed.test_no_lca',
        'evt_mixed_xe_sm80_90.TestEVTMixed.test_mixed_dag',
        'evt_mixed_xe_sm80_90.TestEVTMixed.test_mixed_dag_no_batch',
        'evt_layout_xe_sm80_90.TestEVTLayout.test_permute_1',
        'evt_layout_xe_sm80_90.TestEVTLayout.test_reshape',
        'evt_layout_xe_sm80_90.TestEVTLayout.test_reshape2'
    ],
    'evt_compute': [
        'evt_compute_xe_sm80_90.TestEVTCompute',
    ],
    'evt_load': [
        'evt_load_xe_sm80_90.TestEVTLoad',
    ],
    'evt_store': [
        'evt_store_xe_sm80_90.TestEVTStore',
    ],
    'evt_mixed': [
        'evt_mixed_xe_sm80_90.TestEVTMixed',
    ],
    'evt_layout': [
        'evt_layout_xe_sm80_90.TestEVTLayout',
    ],
    'all': [
        'evt_compute_xe_sm80_90.TestEVTCompute',
        'evt_layout_xe_sm80_90.TestEVTLayout',
        'evt_load_xe_sm80_90.TestEVTLoad',
        'evt_store_xe_sm80_90.TestEVTStore',
        'evt_mixed_xe_sm80_90.TestEVTMixed',
    ],
}


def list_suites():
    """List all available test suites."""
    print("Available test suites:")
    for suite_name in sorted(TEST_SUITES.keys()):
        print(f"  - {suite_name}")
        for test_module in TEST_SUITES[suite_name]:
            print(f"      {test_module}")


def run_test_suite(suite_name, write_csv=False):
    """Run a specific test suite."""
    if suite_name not in TEST_SUITES:
        print(f"Error: Test suite '{suite_name}' not found.")
        print("\nUse --list to see available test suites.")
        return False, None
    
    test_modules = TEST_SUITES[suite_name]
    loader = unittest.TestLoader()
    
    suite = loader.loadTestsFromNames(test_modules)
    testRunner = unittest.TextTestRunner(verbosity=2, resultclass=TestResultWithSuccesses)
    results = testRunner.run(suite)

    # Always print summary
    print_test_summary(results, suite_name)
    
    # Write results to CSV if requested
    if write_csv:
        write_test_results_to_csv(results, suite_name)
    
    return results.wasSuccessful(), results


def main():
    parser = argparse.ArgumentParser(
        description='Run EVT test suites',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_xe_evt_tests.py -j xe_evt_ci
  python run_xe_evt_tests.py --job evt_layout -o
  python run_xe_evt_tests.py --list
  python run_xe_evt_tests.py -j all --output
        """
    )
    
    parser.add_argument(
        '-j', '--job',
        type=str,
        help='Test suite to run (use --list to see available suites)'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available test suites'
    )
    
    parser.add_argument(
        '-o', '--output',
        action='store_true',
        help='Write test results to CSV file (format: test_results_<suite>_<timestamp>.csv)'
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_suites()
        return 0
    
    if not args.job:
        parser.print_help()
        print("\nError: Please specify a test suite using -j/--job or use --list to see available suites.")
        return 1
    
    success, results = run_test_suite(args.job, args.output)
    
    if not success:
        print(f"\nTest suite '{args.job}' failed!")
        return 1
    
    print(f"\nTest suite '{args.job}' passed!")
    return 0


if __name__ == '__main__':
    sys.exit(main())

