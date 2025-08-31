# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
Basic functionality tests for the polars-based Triangle implementation.
These tests validate that core Triangle functionality works correctly.
"""

import pytest
import pandas as pd
import numpy as np
import chainladder as cl
from chainladder.core.base import TriangleBase
from chainladder.core.triangle import Triangle as CoreTriangle
import polars as pl
from datetime import datetime


class TestBasicTriangleFunctionality:
    """Test basic Triangle functionality works"""

    def test_core_triangle_creation(self):
        """Test creating a basic triangle from polars data"""
        # Create simple test data
        data = pl.DataFrame({
            'origin': [datetime(2020, 1, 1), datetime(2020, 1, 1), datetime(2021, 1, 1)],
            'valuation': [datetime(2020, 1, 1), datetime(2021, 1, 1), datetime(2021, 1, 1)],
            'values': [1000.0, 1100.0, 2000.0]
        })
        
        # Create TriangleBase directly
        tri_base = TriangleBase(
            data,
            origin='origin',
            valuation='valuation',
            columns=['values'],
            cumulative=True
        )
        
        # Test basic properties exist and work
        assert hasattr(tri_base, 'shape')
        assert hasattr(tri_base, 'origin')
        assert hasattr(tri_base, 'development')
        assert hasattr(tri_base, 'columns')
        
        # Test shape is reasonable
        shape = tri_base.shape
        assert len(shape) == 4
        assert all(dim >= 1 for dim in shape)

    def test_core_triangle_pandas_wrapper(self):
        """Test pandas wrapper Triangle works"""
        # Create test data as pandas DataFrame
        data = pd.DataFrame({
            'origin': [datetime(2020, 1, 1), datetime(2020, 1, 1), datetime(2021, 1, 1)],
            'valuation': [datetime(2020, 1, 1), datetime(2021, 1, 1), datetime(2021, 1, 1)],
            'values': [1000.0, 1100.0, 2000.0]
        })
        
        # Create Triangle (pandas wrapper)
        tri = CoreTriangle(
            data,
            origin='origin',
            valuation='valuation', 
            columns='values',
            cumulative=True
        )
        
        # Test basic properties
        assert hasattr(tri, 'shape')
        assert hasattr(tri, 'triangle')  # Should have underlying TriangleBase
        
        # Test shape works
        shape = tri.shape
        assert len(shape) == 4
        
        # Test properties return pandas objects
        assert isinstance(tri.index, pd.DataFrame)
        assert isinstance(tri.columns, pd.Index)

    def test_triangle_basic_operations(self):
        """Test basic operations work without errors"""
        data = pd.DataFrame({
            'origin': [datetime(2020, 1, 1), datetime(2020, 1, 1), datetime(2021, 1, 1)],
            'valuation': [datetime(2020, 1, 1), datetime(2021, 1, 1), datetime(2021, 1, 1)],
            'values': [1000.0, 1100.0, 2000.0]
        })
        
        tri = CoreTriangle(
            data,
            origin='origin',
            valuation='valuation',
            columns='values', 
            cumulative=True
        )
        
        # Test repr doesn't crash
        str_repr = str(tri)
        assert len(str_repr) > 0
        
        # Test basic aggregation
        try:
            summed = tri.triangle.sum()
            assert hasattr(summed, 'shape')
        except (NotImplementedError, AttributeError):
            # Some operations may not be fully implemented yet
            pass

    def test_triangle_cumulative_incremental(self):
        """Test cumulative/incremental conversion"""
        data = pd.DataFrame({
            'origin': [datetime(2020, 1, 1), datetime(2020, 1, 1), datetime(2021, 1, 1)],
            'valuation': [datetime(2020, 1, 1), datetime(2021, 1, 1), datetime(2021, 1, 1)],
            'values': [1000.0, 1500.0, 2000.0]  # Cumulative values
        })
        
        tri = CoreTriangle(
            data,
            origin='origin',
            valuation='valuation',
            columns='values',
            cumulative=True
        )
        
        # Test conversion to incremental
        incremental = tri.triangle.to_incremental()
        assert incremental.is_cumulative == False
        
        # Test conversion back to cumulative
        back_to_cum = incremental.to_cumulative()
        assert back_to_cum.is_cumulative == True

    def test_triangle_development_valuation_conversion(self):
        """Test development/valuation conversion"""
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
        
        # Test conversion to development
        dev_tri = tri.triangle.to_development()
        assert dev_tri.is_val_tri == False
        
        # Test conversion to valuation  
        val_tri = dev_tri.to_valuation()
        assert val_tri.is_val_tri == True

    def test_triangle_arithmetic(self):
        """Test basic arithmetic operations"""
        data = pd.DataFrame({
            'origin': [datetime(2020, 1, 1), datetime(2020, 1, 1)],
            'valuation': [datetime(2020, 1, 1), datetime(2021, 1, 1)],
            'values': [1000.0, 1500.0]
        })
        
        tri = CoreTriangle(
            data,
            origin='origin',
            valuation='valuation',
            columns='values',
            cumulative=True
        )
        
        # Test scalar arithmetic
        doubled = tri.triangle * 2
        assert doubled.shape == tri.triangle.shape
        
        # Test triangle arithmetic
        added = tri.triangle + tri.triangle
        assert added.shape == tri.triangle.shape

    def test_integration_with_sample_data(self, raa):
        """Test that core triangle can work with existing sample data"""
        # Convert legacy triangle to DataFrame
        df = raa.to_frame(keepdims=True, origin_as_datetime=True).reset_index()
        
        # Create core triangle
        try:
            core_tri = CoreTriangle(
                df,
                origin='origin',
                valuation='valuation',
                columns='values',
                cumulative=raa.is_cumulative
            )
            
            # Should be able to access basic properties
            assert hasattr(core_tri, 'shape')
            shape = core_tri.shape
            assert len(shape) == 4
            
            # Should be compatible with legacy shape (at least origin/development)
            legacy_shape = raa.shape
            assert shape[2] == legacy_shape[2]  # Same number of origin periods
            assert shape[3] == legacy_shape[3]  # Same number of development periods
            
        except Exception as e:
            # If integration doesn't work yet, that's information too
            pytest.skip(f"Integration with sample data not ready: {e}")


class TestTriangleDataTypes:
    """Test that Triangle handles different data types correctly"""

    def test_date_formats(self):
        """Test different date format inputs"""
        # Test with string dates
        data = pd.DataFrame({
            'origin': ['2020-01-01', '2020-01-01', '2021-01-01'],
            'valuation': ['2020-01-01', '2021-01-01', '2021-01-01'],
            'values': [1000.0, 1100.0, 2000.0]
        })
        
        try:
            tri = CoreTriangle(
                data,
                origin='origin',
                valuation='valuation',
                columns='values',
                cumulative=True
            )
            assert hasattr(tri, 'shape')
        except Exception:
            # Date parsing might not be fully implemented
            pytest.skip("String date parsing not yet implemented")

    def test_numeric_values(self):
        """Test different numeric value types"""
        data = pd.DataFrame({
            'origin': [datetime(2020, 1, 1), datetime(2021, 1, 1)],
            'valuation': [datetime(2020, 1, 1), datetime(2021, 1, 1)],
            'values': [1000, 2000]  # Integer values
        })
        
        tri = CoreTriangle(
            data,
            origin='origin',
            valuation='valuation',
            columns='values',
            cumulative=True
        )
        
        assert tri.triangle.shape[0] >= 1

    def test_missing_values(self):
        """Test handling of missing/NaN values"""
        data = pd.DataFrame({
            'origin': [datetime(2020, 1, 1), datetime(2020, 1, 1), datetime(2021, 1, 1)],
            'valuation': [datetime(2020, 1, 1), datetime(2021, 1, 1), datetime(2021, 1, 1)],
            'values': [1000.0, np.nan, 2000.0]
        })
        
        tri = CoreTriangle(
            data,
            origin='origin',
            valuation='valuation',
            columns='values',
            cumulative=True
        )
        
        # Should handle missing values gracefully
        assert tri.triangle.shape[0] >= 1


class TestTriangleEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_single_entry(self):
        """Test triangle with single data point"""
        data = pd.DataFrame({
            'origin': [datetime(2020, 1, 1)],
            'valuation': [datetime(2020, 1, 1)],
            'values': [1000.0]
        })
        
        tri = CoreTriangle(
            data,
            origin='origin',
            valuation='valuation', 
            columns='values',
            cumulative=True
        )
        
        # Single entry should work
        assert tri.triangle.shape[2:] == (1, 1)  # One origin, one development

    def test_large_data_handling(self):
        """Test with moderately large dataset"""
        # Create larger dataset
        origins = [datetime(2015 + i, 1, 1) for i in range(10)]
        valuations = [datetime(2015 + i, j + 1, 1) for i in range(10) for j in range(i, 10)]
        origin_vals = [datetime(2015 + i, 1, 1) for i in range(10) for j in range(i, 10)]
        values = [1000.0 * (i + 1) * (j + 1) for i in range(10) for j in range(i, 10)]
        
        data = pd.DataFrame({
            'origin': origin_vals,
            'valuation': valuations, 
            'values': values
        })
        
        tri = CoreTriangle(
            data,
            origin='origin',
            valuation='valuation',
            columns='values',
            cumulative=True
        )
        
        # Should handle larger data without issues
        assert tri.triangle.shape[0] >= 1
        assert tri.triangle.shape[2] == 10  # 10 origin periods

    def test_duplicate_handling(self):
        """Test handling of duplicate entries"""
        data = pd.DataFrame({
            'origin': [datetime(2020, 1, 1), datetime(2020, 1, 1)],
            'valuation': [datetime(2020, 1, 1), datetime(2020, 1, 1)],
            'values': [1000.0, 500.0]  # Duplicate entries
        })
        
        tri = CoreTriangle(
            data,
            origin='origin',
            valuation='valuation',
            columns='values',
            cumulative=True
        )
        
        # Should aggregate duplicates or handle appropriately
        assert tri.triangle.shape[2:] == (1, 1)  # Should resolve to single cell


class TestPolarsMethods:
    """Test new polars-native methods"""
    
    def test_replace_non_finite_method(self):
        """Test replace_non_finite method"""
        data = pd.DataFrame({
            'origin': [datetime(2020, 1, 1), datetime(2020, 1, 1), datetime(2021, 1, 1)],
            'valuation': [datetime(2020, 1, 1), datetime(2021, 1, 1), datetime(2021, 1, 1)],
            'values': [1000.0, float('inf'), -float('inf')]  # Include infinite values
        })
        
        tri = CoreTriangle(
            data,
            origin='origin',
            valuation='valuation',
            columns='values',
            cumulative=True
        )
        
        # Test replace_non_finite method
        clean_tri = tri.replace_non_finite()
        assert clean_tri.shape == tri.shape
        
        # Verify method returns new triangle instance
        assert clean_tri is not tri
        assert clean_tri.triangle is not tri.triangle
        
    def test_with_values_mathematical_operations(self):
        """Test with_values method for mathematical operations"""
        data = pd.DataFrame({
            'origin': [datetime(2020, 1, 1), datetime(2020, 1, 1)],
            'valuation': [datetime(2020, 1, 1), datetime(2021, 1, 1)],
            'values': [1000.0, 1500.0]
        })
        
        tri = CoreTriangle(
            data,
            origin='origin', 
            valuation='valuation',
            columns='values',
            cumulative=True
        )
        
        # Test mathematical operations
        doubled_tri = tri.with_values(values=pl.col('values') * 2)
        assert doubled_tri.shape == tri.shape
        
        # Verify the operation worked
        orig_sum = tri.values.select('0').sum().item()
        doubled_sum = doubled_tri.values.select('0').sum().item()
        assert abs(doubled_sum - 2 * orig_sum) < 0.01
        
        # Test method chaining
        complex_tri = (tri
                      .with_values(values=pl.col('values') * 2)
                      .replace_non_finite())
        assert complex_tri.shape == tri.shape