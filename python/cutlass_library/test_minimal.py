#!/usr/bin/env python3
"""
Minimal test to verify BMG kernel generation works correctly
"""

import os
import sys
from pathlib import Path

# Add the cutlass_library to the path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

def minimal_test():
    """Minimal test - just verify generation works"""
    print("\n" + "="*70)
    print("MINIMAL BMG GENERATION TEST")
    print("="*70)
    
    from generator import GenerateBMG
    from manifest import Manifest
    
    print("\nStep 1: Creating manifest for BMG...")
    
    try:
        class Args:
            operations = 'gemm'
            build_dir = './minimal_test_build'
            curr_build_dir = './minimal_test_build'
            architectures = 'bmg'  # Intel BMG/Xe2
            kernel_filter_file = None
            selected_kernel_list = None
            interface_dir = None
            filter_by_cc = True
            kernels = ''
            ignore_kernels = ''
            exclude_kernels = ''
            cuda_version = '12.0'
            disable_full_archs_compilation = False
            instantiation_level = '0'
        
        manifest = Manifest(Args())
        print(f"✓ Manifest created")
        print(f"  - Compute capabilities: {manifest.compute_capabilities_baseline}")
        print(f"  - Is Xe target: {manifest.is_xe_target}")
        
        if not manifest.is_xe_target:
            print("✗ FAIL: is_xe_target should be True!")
            return False
            
        if 20 not in manifest.compute_capabilities_baseline:
            print("✗ FAIL: Architecture 20 not in baseline!")
            return False
            
    except Exception as e:
        print(f"✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\nStep 2: Generating BMG operations...")
    
    try:
        GenerateBMG(manifest, '12.0')
        
        op_count = manifest.operation_count
        print(f"✓ Generated {op_count} operations")
        
        if op_count == 0:
            print("✗ FAIL: No operations generated!")
            return False
            
    except Exception as e:
        print(f"✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\nStep 3: Verifying operations were added to manifest...")
    
    try:
        # Just verify operations exist
        from library import OperationKind
        if OperationKind.Gemm in manifest.operations:
            print(f"✓ GEMM operations added to manifest")
            print(f"  - {len(manifest.operations[OperationKind.Gemm])} operation configurations")
        else:
            print("✗ FAIL: GEMM operation kind not in manifest")
            return False
            
    except Exception as e:
        print(f"✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\nStep 4: Testing file extension logic...")
    
    try:
        from gemm_operation import EmitGemmConfigurationLibrary
        from pathlib import Path as P
        
        # Test Xe architecture path (with xe prefix as it would be generated)
        test_path = P("./test_temp/gemm/20/xe20_dpas")
        test_path.mkdir(parents=True, exist_ok=True)
        
        emitter = EmitGemmConfigurationLibrary(str(test_path), "test_config")
        ext = P(emitter.configuration_path).suffix
        
        print(f"  - Intel Xe (xe20 path) file extension: {ext}")
        
        if ext != ".cpp":
            print(f"✗ FAIL: Expected .cpp extension, got {ext}")
            import shutil
            shutil.rmtree("./test_temp")
            return False
            
        print("✓ File extension correct (.cpp for Intel Xe)")
        
        # Test CUDA path for comparison
        test_path_cuda = P("./test_temp/gemm/90/sm90_tensorop")
        test_path_cuda.mkdir(parents=True, exist_ok=True)
        
        emitter_cuda = EmitGemmConfigurationLibrary(str(test_path_cuda), "test_cuda_config")
        ext_cuda = P(emitter_cuda.configuration_path).suffix
        
        print(f"  - CUDA (sm90 path) file extension: {ext_cuda}")
        
        if ext_cuda != ".cu":
            print(f"✗ FAIL: Expected .cu extension for CUDA, got {ext_cuda}")
            import shutil
            shutil.rmtree("./test_temp")
            return False
            
        print("✓ File extension correct (.cu for CUDA)")
        
        # Clean up
        import shutil
        shutil.rmtree("./test_temp")
        
    except Exception as e:
        print(f"✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED!")
    print("="*70)
    print(f"\nSummary:")
    print(f"  - Generated {op_count} BMG operations")
    print(f"  - Architecture 20 (BMG/Xe2) correctly detected")
    print(f"  - File extension .cpp (not .cu) for Intel Xe")
    print(f"  - is_xe flag correctly set")
    
    return True


if __name__ == "__main__":
    success = minimal_test()
    sys.exit(0 if success else 1)
