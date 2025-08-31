#!/usr/bin/env python3
"""
Simple test execution for the triangle implementation
"""

import sys
import os
import subprocess
import tempfile

def run_pytest_test(test_file, test_method=None):
    """Run a specific pytest test"""
    cmd = [sys.executable, "-m", "pytest", test_file, "-v"]
    if test_method:
        cmd.extend(["-k", test_method])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)

def main():
    """Run the tests and report results"""
    print("=== Running Triangle Tests ===\n")
    
    # List of test files to run
    test_files = [
        "chainladder/core/tests/test_basic_functionality.py",
        "chainladder/core/tests/test_core_legacy_parity.py", 
        "chainladder/core/tests/test_triangle_public_api.py"
    ]
    
    all_results = []
    
    for test_file in test_files:
        print(f"\n--- Running {test_file} ---")
        returncode, stdout, stderr = run_pytest_test(test_file)
        
        print(f"Return code: {returncode}")
        if stdout:
            print("STDOUT:")
            print(stdout)
        if stderr:
            print("STDERR:")
            print(stderr)
        
        all_results.append((test_file, returncode, stdout, stderr))
        print("-" * 60)
    
    # Summary
    print("\n=== SUMMARY ===")
    passed = 0
    total = len(test_files)
    
    for test_file, returncode, stdout, stderr in all_results:
        status = "PASSED" if returncode == 0 else "FAILED"
        if returncode == 0:
            passed += 1
        print(f"{test_file}: {status}")
    
    print(f"\nOverall: {passed}/{total} test files passed")
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)