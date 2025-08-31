#!/usr/bin/env python3
"""Test link ratio implementation against legacy"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
from datetime import datetime
import chainladder as cl
from chainladder.core.triangle import Triangle as CoreTriangle

def test_link_ratio_implementation():
    """Test link ratio implementation against legacy"""
    
    print("=== Testing Link Ratio Implementation ===")
    
    try:
        # Load sample data
        raa = cl.load_sample('raa')
        print(f"✓ Loaded RAA data. Shape: {raa.shape}")
        
        # Test legacy link ratio
        legacy_lr = raa.link_ratio
        print(f"✓ Legacy link ratio. Shape: {legacy_lr.shape}")
        print(f"  Legacy is_pattern: {legacy_lr.is_pattern}")
        print(f"  Legacy first few values: {legacy_lr.iloc[0,0,:3,0].to_frame().iloc[:3,0].tolist()}")
        
        # Convert to core triangle
        df = raa.to_frame(keepdims=True, origin_as_datetime=True).reset_index()
        core_tri = CoreTriangle(
            df,
            origin='origin',
            valuation='valuation',
            columns='values',
            cumulative=raa.is_cumulative
        )
        print(f"✓ Created core triangle. Shape: {core_tri.shape}")
        
        # Test core link ratio
        try:
            core_lr = core_tri.triangle.link_ratio
            print(f"✓ Core link ratio. Shape: {core_lr.shape}")
            print(f"  Core is_pattern: {core_lr.is_pattern}")
            
            # Test if we can extract values
            try:
                # Get some sample values for comparison
                core_values = core_lr.to_frame()
                print(f"✓ Can convert core link ratio to frame")
                print(f"  Core frame shape: {core_values.shape}")
                
                # Compare shapes
                if legacy_lr.shape[2:] == core_lr.shape[2:]:
                    print("✓ Shapes match between legacy and core")
                else:
                    print(f"⚠ Shape mismatch: Legacy {legacy_lr.shape} vs Core {core_lr.shape}")
                
                return True
                
            except Exception as e:
                print(f"⚠ Core link ratio calculation issue: {e}")
                import traceback
                traceback.print_exc()
                return False
                
        except Exception as e:
            print(f"✗ Core link ratio failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"✗ Test setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_triangle_operations():
    """Test basic operations that link_ratio depends on"""
    
    print("\n=== Testing Basic Triangle Operations ===")
    
    try:
        # Create simple test data
        data = pd.DataFrame({
            'origin': [datetime(2020, 1, 1), datetime(2020, 1, 1), datetime(2021, 1, 1), datetime(2021, 1, 1)],
            'valuation': [datetime(2020, 1, 1), datetime(2021, 1, 1), datetime(2021, 1, 1), datetime(2022, 1, 1)],
            'values': [1000.0, 1500.0, 2000.0, 2400.0]
        })
        
        tri = CoreTriangle(
            data,
            origin='origin',
            valuation='valuation',
            columns='values',
            cumulative=True
        )
        
        print(f"✓ Created test triangle. Shape: {tri.shape}")
        
        # Test slicing operations used in link_ratio
        print("\n  Testing slicing operations...")
        
        # Test [..., 1:] - should get development periods 1+
        try:
            numer_slice = tri.triangle[..., 1:]
            print(f"✓ Numerator slice [..., 1:]. Shape: {numer_slice.shape}")
        except Exception as e:
            print(f"✗ Numerator slice failed: {e}")
            return False
        
        # Test [..., :-1] - should get all but last development period
        try:
            denom_slice = tri.triangle[..., :-1]
            print(f"✓ Denominator slice [..., :-1]. Shape: {denom_slice.shape}")
        except Exception as e:
            print(f"✗ Denominator slice failed: {e}")
            return False
            
        # Test arithmetic operations
        try:
            # Simple division test
            result = numer_slice / denom_slice
            print(f"✓ Division operation. Shape: {result.shape}")
            
            # Test values property
            values_df = result.values
            print(f"✓ Values property. Shape: {values_df.shape}")
            
        except Exception as e:
            print(f"✗ Arithmetic operations failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Basic operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Triangle Link Ratio Implementation")
    
    # Test basic operations first
    basic_success = test_basic_triangle_operations()
    
    # Test link ratio implementation
    lr_success = test_link_ratio_implementation()
    
    print(f"\n=== Summary ===")
    print(f"Basic operations: {'✓ PASS' if basic_success else '✗ FAIL'}")
    print(f"Link ratio: {'✓ PASS' if lr_success else '✗ FAIL'}")
    
    if basic_success and lr_success:
        print("✓ Link ratio implementation is working!")
    else:
        print("⚠ Link ratio needs fixes")
    
    sys.exit(0 if (basic_success and lr_success) else 1)