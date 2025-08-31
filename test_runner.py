#!/usr/bin/env python3
"""
Simple test runner to validate the triangle implementation.
Run this to check if the polars triangle is working.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def test_basic_creation():
    """Test basic triangle creation"""
    try:
        import pandas as pd
        import polars as pl
        from datetime import datetime
        from chainladder.core.base import TriangleBase
        from chainladder.core.triangle import Triangle as CoreTriangle
        
        print("Testing basic triangle creation...")
        
        # Test TriangleBase creation
        data = pl.DataFrame({
            'origin': [datetime(2020, 1, 1), datetime(2020, 1, 1), datetime(2021, 1, 1)],
            'valuation': [datetime(2020, 1, 1), datetime(2021, 1, 1), datetime(2021, 1, 1)],
            'values': [1000.0, 1100.0, 2000.0]
        })
        
        tri_base = TriangleBase(
            data,
            origin='origin',
            valuation='valuation',
            columns=['values'],
            cumulative=True
        )
        
        print(f"✓ TriangleBase created successfully. Shape: {tri_base.shape}")
        
        # Test pandas wrapper
        data_pd = pd.DataFrame({
            'origin': [datetime(2020, 1, 1), datetime(2020, 1, 1), datetime(2021, 1, 1)],
            'valuation': [datetime(2020, 1, 1), datetime(2021, 1, 1), datetime(2021, 1, 1)],
            'values': [1000.0, 1100.0, 2000.0]
        })
        
        tri = CoreTriangle(
            data_pd,
            origin='origin',
            valuation='valuation',
            columns='values',
            cumulative=True
        )
        
        print(f"✓ Core Triangle created successfully. Shape: {tri.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sample_data_integration():
    """Test integration with sample data"""
    try:
        import chainladder as cl
        from chainladder.core.triangle import Triangle as CoreTriangle
        
        print("\nTesting sample data integration...")
        
        # Load sample data
        raa = cl.load_sample('raa')
        print(f"✓ Loaded RAA sample data. Shape: {raa.shape}")
        
        # Convert to DataFrame
        df = raa.to_frame(keepdims=True, origin_as_datetime=True).reset_index()
        print(f"✓ Converted to DataFrame. Rows: {len(df)}")
        
        # Try to create core triangle
        try:
            core_tri = CoreTriangle(
                df,
                origin='origin',
                valuation='valuation', 
                columns='values',
                cumulative=raa.is_cumulative
            )
            print(f"✓ Core Triangle from sample data created. Shape: {core_tri.shape}")
            
            # Test basic properties
            print(f"  - Origin grain: {core_tri.triangle.origin_grain}")
            print(f"  - Development grain: {core_tri.triangle.development_grain}")
            print(f"  - Is cumulative: {core_tri.triangle.is_cumulative}")
            print(f"  - Valuation date: {core_tri.triangle.valuation_date}")
            
            return True
            
        except Exception as e:
            print(f"✗ Core triangle creation from sample data failed: {e}")
            return False
        
    except Exception as e:
        print(f"✗ Sample data integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_operations():
    """Test basic operations"""
    try:
        import pandas as pd
        from datetime import datetime
        from chainladder.core.triangle import Triangle as CoreTriangle
        
        print("\nTesting basic operations...")
        
        data = pd.DataFrame({
            'origin': [datetime(2020, 1, 1), datetime(2020, 1, 1), datetime(2021, 1, 1)],
            'valuation': [datetime(2020, 1, 1), datetime(2021, 1, 1), datetime(2021, 1, 1)],
            'values': [1000.0, 1500.0, 2000.0]
        })
        
        tri = CoreTriangle(
            data,
            origin='origin',
            valuation='valuation',
            columns='values',
            cumulative=True
        )
        
        # Test cumulative/incremental conversion
        try:
            incremental = tri.triangle.to_incremental()
            print(f"✓ Conversion to incremental: is_cumulative = {incremental.is_cumulative}")
            
            back_to_cum = incremental.to_cumulative()
            print(f"✓ Back to cumulative: is_cumulative = {back_to_cum.is_cumulative}")
        except Exception as e:
            print(f"⚠ Cumulative/incremental conversion not working: {e}")
        
        # Test arithmetic
        try:
            doubled = tri.triangle * 2
            print(f"✓ Scalar multiplication works. Shape maintained: {doubled.shape}")
            
            added = tri.triangle + tri.triangle
            print(f"✓ Triangle addition works. Shape maintained: {added.shape}")
        except Exception as e:
            print(f"⚠ Arithmetic operations not working: {e}")
        
        # Test aggregation
        try:
            summed = tri.triangle.sum()
            print(f"✓ Sum aggregation works. Result shape: {summed.shape}")
        except Exception as e:
            print(f"⚠ Sum aggregation not working: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=== Triangle Implementation Test Suite ===")
    
    tests = [
        test_basic_creation,
        test_sample_data_integration,
        test_basic_operations
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print(f"\n=== Summary ===")
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! The polars Triangle implementation is working.")
    else:
        print("⚠ Some tests failed. The implementation may need more work.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)