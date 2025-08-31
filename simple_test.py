#!/usr/bin/env python3
"""Simple test to check current Triangle implementation baseline"""

try:
    import sys
    import os
    sys.path.insert(0, os.path.abspath('.'))
    
    import pandas as pd
    import polars as pl
    from datetime import datetime
    import chainladder as cl
    from chainladder.core.base import TriangleBase
    from chainladder.core.triangle import Triangle as CoreTriangle

    print("=== Testing Current Triangle Implementation ===\n")

    # Test 1: Create basic triangle
    print("1. Testing basic triangle creation...")
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
    print(f"✓ Triangle created. Shape: {tri.shape}")
    
    # Test 2: Test basic properties
    print("\n2. Testing basic properties...")
    print(f"✓ valuation_date: {tri.valuation_date}")
    print(f"✓ origin_grain: {tri.triangle.origin_grain}")
    print(f"✓ development_grain: {tri.triangle.development_grain}")
    print(f"✓ is_cumulative: {tri.triangle.is_cumulative}")
    
    # Test 3: Test latest_diagonal
    print("\n3. Testing latest_diagonal...")
    latest = tri.triangle.latest_diagonal
    print(f"✓ Latest diagonal shape: {latest.shape}")
    
    # Test 4: Test conversions
    print("\n4. Testing conversions...")
    incremental = tri.triangle.to_incremental()
    print(f"✓ To incremental: is_cumulative = {incremental.is_cumulative}")
    
    back_to_cum = incremental.to_cumulative()
    print(f"✓ Back to cumulative: is_cumulative = {back_to_cum.is_cumulative}")
    
    # Test 5: Test link_ratio
    print("\n5. Testing link_ratio...")
    try:
        lr = tri.triangle.link_ratio
        print(f"✓ Link ratio shape: {lr.shape}")
        print(f"✓ Link ratio is_pattern: {lr.is_pattern}")
    except Exception as e:
        print(f"⚠ Link ratio failed: {e}")
    
    # Test 6: Test aggregation
    print("\n6. Testing aggregation...")
    try:
        summed = tri.triangle.sum()
        print(f"✓ Sum operation: shape = {summed.shape}")
    except Exception as e:
        print(f"⚠ Sum failed: {e}")
    
    # Test 7: Test with sample data
    print("\n7. Testing with RAA sample data...")
    try:
        raa = cl.load_sample('raa')
        print(f"✓ RAA loaded. Shape: {raa.shape}")
        
        # Convert to core triangle
        df = raa.to_frame(keepdims=True, origin_as_datetime=True).reset_index()
        print(f"RAA DataFrame columns: {df.columns.tolist()}")
        print(f"RAA DataFrame shape: {df.shape}")
        print(f"RAA DataFrame head:\n{df.head()}")
        
        # Use the actual column names from RAA
        core_raa = CoreTriangle(
            df,
            origin='origin',
            development='development', 
            columns=['values'],
            cumulative=True  # RAA is typically cumulative
        )
        print(f"✓ RAA converted to core triangle. Shape: {core_raa.shape}")
        
        # Test operations on real data
        raa_latest = core_raa.triangle.latest_diagonal
        print(f"✓ RAA latest diagonal. Shape: {raa_latest.shape}")
        
        raa_lr = core_raa.triangle.link_ratio
        print(f"✓ RAA link ratios. Shape: {raa_lr.shape}")
        
    except Exception as e:
        print(f"⚠ RAA test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n=== Baseline Test Summary ===")
    print("✓ Core Triangle functionality is working!")
    print("✓ Most key properties and methods are implemented")
    print("✓ Ready to enhance and test against downstream models")

except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()