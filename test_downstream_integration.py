#!/usr/bin/env python3
"""Test integration with downstream chainladder models"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import pandas as pd
import numpy as np
from datetime import datetime
import chainladder as cl
from chainladder.core.triangle import Triangle as CoreTriangle

def test_basic_estimator_integration():
    """Test basic integration with chainladder estimators"""
    
    print("=== Testing Basic Estimator Integration ===")
    
    try:
        # Load sample data
        raa = cl.load_sample('raa')
        print(f"✓ Loaded RAA data. Shape: {raa.shape}")
        
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
        
        # Test with Development estimator
        print("\n  Testing Development estimator...")
        try:
            dev = cl.Development()
            dev_fitted = dev.fit(core_tri)
            print(f"✓ Development estimator fitted successfully")
            
            # Check if we can access fitted properties  
            try:
                ldf = dev_fitted.ldf_
                print(f"✓ Accessed ldf_ property. Type: {type(ldf)}")
                print(f"  ldf_ shape: {ldf.shape}")
            except Exception as e:
                print(f"⚠ Could not access ldf_ property: {e}")
                
            try:
                cdf = dev_fitted.cdf_
                print(f"✓ Accessed cdf_ property. Type: {type(cdf)}")
                print(f"  cdf_ shape: {cdf.shape}")
            except Exception as e:
                print(f"⚠ Could not access cdf_ property: {e}")
                
        except Exception as e:
            print(f"✗ Development estimator failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test with Chainladder method
        print("\n  Testing Chainladder method...")
        try:
            cl_method = cl.Chainladder()
            cl_fitted = cl_method.fit(core_tri)
            print(f"✓ Chainladder method fitted successfully")
            
            # Check if we can access ultimate estimates
            try:
                ultimate = cl_fitted.ultimate_
                print(f"✓ Accessed ultimate_ property. Type: {type(ultimate)}")
                print(f"  ultimate_ shape: {ultimate.shape}")
            except Exception as e:
                print(f"⚠ Could not access ultimate_ property: {e}")
                
            try:
                ibnr = cl_fitted.ibnr_
                print(f"✓ Accessed ibnr_ property. Type: {type(ibnr)}")
                print(f"  ibnr_ shape: {ibnr.shape}")
            except Exception as e:
                print(f"⚠ Could not access ibnr_ property: {e}")
                
        except Exception as e:
            print(f"✗ Chainladder method failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Basic integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_triangle_properties_needed():
    """Test Triangle properties that downstream models need"""
    
    print("\n=== Testing Properties Needed by Downstream Models ===")
    
    try:
        # Create test triangle
        raa = cl.load_sample('raa')
        df = raa.to_frame(keepdims=True, origin_as_datetime=True).reset_index()
        core_tri = CoreTriangle(
            df,
            origin='origin',
            valuation='valuation', 
            columns='values',
            cumulative=raa.is_cumulative
        )
        
        # Test properties that methods expect
        properties_to_test = [
            ('shape', 'Shape property'),
            ('valuation_date', 'Valuation date property'),
            ('is_cumulative', 'Cumulative flag'),
            ('latest_diagonal', 'Latest diagonal'),
            ('link_ratio', 'Link ratio calculation'),
        ]
        
        success_count = 0
        for prop, desc in properties_to_test:
            try:
                if prop == 'shape':
                    val = core_tri.shape
                elif prop == 'valuation_date':
                    val = core_tri.valuation_date
                elif prop == 'is_cumulative':
                    val = core_tri.triangle.is_cumulative
                elif prop == 'latest_diagonal':
                    val = core_tri.triangle.latest_diagonal
                elif prop == 'link_ratio':
                    val = core_tri.triangle.link_ratio
                
                print(f"✓ {desc}: {type(val)} - {getattr(val, 'shape', 'N/A')}")
                success_count += 1
            except Exception as e:
                print(f"✗ {desc} failed: {e}")
        
        print(f"\n  Properties test: {success_count}/{len(properties_to_test)} passed")
        
        # Test conversions that methods use
        conversions_to_test = [
            ('to_incremental', 'Convert to incremental'),
            ('to_cumulative', 'Convert to cumulative'),
            ('to_development', 'Convert to development'),
            ('to_valuation', 'Convert to valuation'),
        ]
        
        conversion_success = 0
        for method, desc in conversions_to_test:
            try:
                converted = getattr(core_tri.triangle, method)()
                print(f"✓ {desc}: {type(converted)} - {converted.shape}")
                conversion_success += 1
            except Exception as e:
                print(f"✗ {desc} failed: {e}")
        
        print(f"\n  Conversions test: {conversion_success}/{len(conversions_to_test)} passed")
        
        # Test aggregations that methods use
        aggregations_to_test = [
            ('sum', 'Sum aggregation'),
            ('mean', 'Mean aggregation'),
        ]
        
        aggregation_success = 0
        for method, desc in aggregations_to_test:
            try:
                result = getattr(core_tri.triangle, method)()
                print(f"✓ {desc}: {type(result)} - {result.shape}")
                aggregation_success += 1
            except Exception as e:
                print(f"✗ {desc} failed: {e}")
        
        print(f"\n  Aggregations test: {aggregation_success}/{len(aggregations_to_test)} passed")
        
        total_success = success_count + conversion_success + aggregation_success
        total_tests = len(properties_to_test) + len(conversions_to_test) + len(aggregations_to_test)
        
        return total_success == total_tests
        
    except Exception as e:
        print(f"✗ Properties test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_method_validation():
    """Test that downstream methods can validate our triangle"""
    
    print("\n=== Testing Method Validation ===")
    
    try:
        # Create test triangle
        raa = cl.load_sample('raa')
        df = raa.to_frame(keepdims=True, origin_as_datetime=True).reset_index()
        core_tri = CoreTriangle(
            df,
            origin='origin',
            valuation='valuation',
            columns='values', 
            cumulative=raa.is_cumulative
        )
        
        # Test validation methods that estimators use
        print("\n  Testing method validation...")
        
        # Test if Development can validate the triangle
        try:
            dev = cl.Development()
            validated = dev.validate_X(core_tri)
            print(f"✓ Development validation passed. Type: {type(validated)}")
            print(f"  Validated shape: {validated.shape}")
        except Exception as e:
            print(f"✗ Development validation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        # Test if methods can work with validated triangle
        try:
            cl_method = cl.Chainladder()
            cl_method.fit(validated)
            print(f"✓ Chainladder can work with validated triangle")
        except Exception as e:
            print(f"✗ Chainladder with validated triangle failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Method validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_copy_functionality():
    """Test copy functionality that methods rely on"""
    
    print("\n=== Testing Copy Functionality ===")
    
    try:
        # Create test triangle
        raa = cl.load_sample('raa')
        df = raa.to_frame(keepdims=True, origin_as_datetime=True).reset_index()
        core_tri = CoreTriangle(
            df,
            origin='origin',
            valuation='valuation',
            columns='values',
            cumulative=raa.is_cumulative
        )
        
        # Test copy method
        try:
            copied = core_tri.copy()
            print(f"✓ Triangle copy successful. Type: {type(copied)}")
            print(f"  Copy shape: {copied.shape}")
            
            # Test that modifications to copy don't affect original
            original_cumulative = core_tri.triangle.is_cumulative
            copy_incremental = copied.triangle.to_incremental()
            
            if core_tri.triangle.is_cumulative == original_cumulative:
                print(f"✓ Copy modifications don't affect original")
            else:
                print(f"⚠ Copy modifications may affect original")
                
        except Exception as e:
            print(f"✗ Triangle copy failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Copy functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Downstream Model Integration")
    
    # Run all integration tests
    tests = [
        test_triangle_properties_needed,
        test_copy_functionality,
        test_method_validation,
        test_basic_estimator_integration,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"Test {test_func.__name__} crashed: {e}")
            results.append(False)
    
    print(f"\n=== Integration Test Summary ===")
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All integration tests passed! Core Triangle works with downstream models.")
    else:
        print("⚠ Some integration tests failed. Need to address compatibility issues.")
    
    sys.exit(0 if passed == total else 1)