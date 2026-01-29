#################################################################################################
# Copyright (C) 2025 Intel Corporation, All rights reserved.
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

import csv
from datetime import datetime
import unittest


class TestResultWithSuccesses(unittest.TextTestResult):
    """Custom TestResult that tracks successful tests."""
    
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.successes = []
    
    def addSuccess(self, test):
        super().addSuccess(test)
        self.successes.append(test)


def print_test_summary(test_results, suite_name):
    """
    Print test results summary.
    
    Args:
        test_results: unittest.TestResult object containing test results
        suite_name: Name of the test suite that was run
    """
    # Check if test_results has successes attribute (custom result class)
    if hasattr(test_results, 'successes'):
        num_passed = len(test_results.successes)
    else:
        # Calculate successful tests (total - failures - errors - skipped)
        num_passed = (test_results.testsRun - len(test_results.failures) - 
                      len(test_results.errors) - len(test_results.skipped))
    
    num_failures = len(test_results.failures)
    num_errors = len(test_results.errors)
    num_skipped = len(test_results.skipped)
    total = test_results.testsRun
    
    print(f"\n{'='*70}")
    print(f"Test Report Summary")
    print(f"{'='*70}")
    print(f"Suite: {suite_name}")
    print(f"Total tests run: {total}")
    print(f"Passed: {num_passed}")
    print(f"Failed: {num_failures}")
    print(f"Errors: {num_errors}")
    print(f"Skipped: {num_skipped}")
    print(f"{'='*70}\n")


def write_test_results_to_csv(test_results, suite_name):
    """
    Write test results to a CSV file.
    
    Args:
        test_results: unittest.TestResult object containing test results
        suite_name: Name of the test suite that was run
    
    Returns:
        str: Path to the generated CSV file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"test_results_{suite_name}_{timestamp}.csv"
    
    # Collect all test results
    all_tests = []
    
    # Check if test_results has successes attribute (custom result class)
    if hasattr(test_results, 'successes'):
        # Add successful tests
        for test in test_results.successes:
            test_name = str(test)
            all_tests.append({
                'test_name': test_name,
                'status': 'PASS',
                'message': ''
            })
    
    # Add failed tests
    for test, traceback in test_results.failures:
        test_name = str(test)
        # Extract just the error message (first line of traceback)
        message = traceback.split('\n')[-2] if traceback else 'Failed'
        all_tests.append({
            'test_name': test_name,
            'status': 'FAIL',
            'message': message
        })
    
    # Add error tests
    for test, traceback in test_results.errors:
        test_name = str(test)
        # Extract just the error message (first line of traceback)
        message = traceback.split('\n')[-2] if traceback else 'Error'
        all_tests.append({
            'test_name': test_name,
            'status': 'ERROR',
            'message': message
        })
    
    # Add skipped tests
    for test, reason in test_results.skipped:
        test_name = str(test)
        all_tests.append({
            'test_name': test_name,
            'status': 'SKIP',
            'message': reason
        })
    
    # Write to CSV
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['test_name', 'status', 'message']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for test_result in all_tests:
            writer.writerow(test_result)
    
    print(f"Results written to: {output_file}\n")
    
    return output_file

