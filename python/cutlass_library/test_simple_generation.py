#!/usr/bin/env python3
"""
Simple test script to generate a small set of BMG kernels
and verify the output files have correct extensions.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the cutlass_library to the path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

def simple_generation_test(build_dir, architecture='20'):
    """
    Simple test that mimics what CMake does
    
    :param build_dir: Directory to output generated files
    :param architecture: Architecture to generate for - supports:
                        - '20', 'bmg', 'xe2' for BMG/Battlemage
                        - '12', 'pvc' for PVC/Ponte Vecchio
    """
    print("\n" + "="*70)
    print("SIMPLE KERNEL GENERATION TEST")
    print("="*70)
    
    # Import after adding to path
    from generator import GenerateIntelXe
    from manifest import Manifest
    from library import OperationKind
    
    # Determine expected architecture number
    arch_map = {
        '20': 20, 'bmg': 20, 'xe2': 20, 'intel_gpu_bmg_g21': 20,
        '12': 12, 'pvc': 12, 'intel_gpu_pvc': 12
    }
    
    arch_lower = architecture.lower()
    if arch_lower not in arch_map:
        print(f"✗ ERROR: Unknown architecture '{architecture}'")
        print(f"  Supported: {list(arch_map.keys())}")
        return False
    
    expected_arch = arch_map[arch_lower]
    arch_name = "BMG/Xe2" if expected_arch == 20 else "PVC"
    
    build_path = Path(build_dir)
    build_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nBuild directory: {build_path}")
    print(f"Architecture: {arch_name} (arch {expected_arch})")
    
    print("\nStep 1: Creating manifest...")
    
    try:
        # Create manifest first (needed by generator)
        class Args:
            operations = 'gemm'
            build_dir = str(build_path)
            curr_build_dir = str(build_path)
            architectures = architecture  # Use provided architecture
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
            print("✗ ERROR: is_xe_target should be True!")
            return False
            
        if expected_arch not in manifest.compute_capabilities_baseline:
            print(f"✗ ERROR: Architecture {expected_arch} not in baseline!")
            return False
            
    except Exception as e:
        print(f"✗ ERROR: Failed to create manifest: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\nStep 2: Generating {arch_name} operations...")
    
    try:
        # Generate operations (adds them to manifest)
        GenerateIntelXe(manifest, '12.0', arch=expected_arch)
        
        # Check operation count
        op_count = manifest.operation_count
        print(f"✓ Generated {op_count} operations")
        
        if op_count == 0:
            print("✗ ERROR: No operations generated!")
            return False
            
    except Exception as e:
        print(f"✗ ERROR: Failed to generate operations: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\nStep 3: Generating library files...")
    
    try:
        # Generate the actual library files
        from library import OperationKind, OperationKindNames, GeneratorTarget
        
        generated_path = build_path / "tools" / "library" / "generated"
        
        # Emit all generated operations (using GeneratorTarget.Library)
        print(f"  - Emitting operations...")
        manifest.emit(GeneratorTarget.Library)
        
        print(f"✓ Library files generated")
        
    except Exception as e:
        print(f"✗ ERROR: Failed to generate library files: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\nStep 4: Verifying generated files...")
    
    # Check for .cpp files in the actual generated directory
    # The manifest creates files in curr_build_dir/generated, not curr_build_dir/tools/library/generated
    actual_generated_path = build_path / "generated"
    gemm_dir = actual_generated_path / "gemm" / str(expected_arch)
    
    if not gemm_dir.exists():
        print(f"✗ ERROR: Directory not created: {gemm_dir}")
        return False
    
    print(f"✓ Directory created: {gemm_dir}")
    
    # Count files
    cpp_files = list(gemm_dir.rglob("*.cpp"))
    cu_files = list(gemm_dir.rglob("*.cu"))
    
    print(f"\n  Generated files:")
    print(f"    - .cpp files: {len(cpp_files)}")
    print(f"    - .cu files:  {len(cu_files)}")
    
    if len(cpp_files) == 0:
        print("✗ ERROR: No .cpp files generated!")
        return False
    
    if len(cu_files) > 0:
        print(f"✗ ERROR: Found {len(cu_files)} .cu files (should be 0 for Intel Xe)!")
        print("  Files:")
        for f in cu_files:
            print(f"    - {f}")
        return False
    
    print("\n  Sample generated files:")
    for cpp_file in cpp_files[:5]:
        print(f"    ✓ {cpp_file.name}")
    
    print("\n" + "="*70)
    print("✓ TEST PASSED - All files generated with .cpp extension!")
    print("="*70)
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple kernel generation test")
    parser.add_argument(
        "--build-dir", "-b",
        default="./test_simple_build",
        help="Build directory (default: ./test_simple_build)"
    )
    parser.add_argument(
        "--arch", "-a",
        default="20",
        help="Architecture to generate for: 20/bmg/xe2 (BMG) or 12/pvc (PVC) (default: 20)"
    )
    
    args = parser.parse_args()
    
    success = simple_generation_test(args.build_dir, args.arch)
    sys.exit(0 if success else 1)
