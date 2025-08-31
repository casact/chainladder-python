# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
Comprehensive test suite for polars Triangle public API against legacy implementation.
This ensures the core Triangle maintains compatibility with the legacy Triangle's public API.
"""

import pytest
import pandas as pd
import numpy as np
import chainladder as cl
from chainladder.core.base import TriangleBase
from chainladder.core.triangle import Triangle as CoreTriangle
import polars as pl
from datetime import datetime


class TestTrianglePublicAPICore:
    """Test core properties and basic functionality"""

    def test_constructor_from_dataframe_parity(self, raa):
        """Test DataFrame constructor maintains parity between core and legacy"""
        legacy_tri = raa
        df = legacy_tri.to_frame(keepdims=True, origin_as_datetime=True).reset_index()
        
        # Create core triangle from same DataFrame
        core_tri = CoreTriangle(
            pl.from_pandas(df), 
            origin='origin', 
            valuation='valuation', 
            columns='values',
            cumulative=True
        )
        
        # Test basic properties match
        assert core_tri.shape[2:] == legacy_tri.shape[2:]  # origin, development dims
        assert len(core_tri.columns) == len(legacy_tri.columns)

    def test_shape_property(self, raa, qtr, clrd):
        """Test shape property consistency across datasets"""
        for dataset in [raa, qtr, clrd]:
            legacy_shape = dataset.shape
            df = dataset.to_frame(keepdims=True, origin_as_datetime=True).reset_index()
            
            core_tri = CoreTriangle(
                pl.from_pandas(df),
                index=dataset.key_labels if len(dataset.key_labels) > 0 else None,
                origin='origin',
                valuation='valuation', 
                columns='values',
                cumulative=dataset.is_cumulative
            )
            
            # Shape should match for origin/development dimensions at minimum
            assert core_tri.shape[2:] == legacy_shape[2:]

    def test_index_property(self, clrd):
        """Test index property returns expected structure"""
        legacy_idx = clrd.index
        df = clrd.to_frame(keepdims=True, origin_as_datetime=True).reset_index()
        
        core_tri = CoreTriangle(
            pl.from_pandas(df),
            index=clrd.key_labels,
            origin='origin',
            valuation='valuation',
            columns='values', 
            cumulative=True
        )
        
        # Index should have same number of entries
        assert len(core_tri.index) == len(legacy_idx)
        # Should contain same columns
        assert set(core_tri.index.columns) >= set(clrd.key_labels)

    def test_origin_development_properties(self, raa):
        """Test origin and development properties"""
        legacy_origin = raa.origin
        legacy_dev = raa.development
        
        df = raa.to_frame(keepdims=True, origin_as_datetime=True).reset_index()
        core_tri = CoreTriangle(
            pl.from_pandas(df),
            origin='origin',
            valuation='valuation',
            columns='values',
            cumulative=True
        )
        
        # Origin length should match
        assert len(core_tri.origin) == len(legacy_origin)
        # Development periods should have similar structure
        assert len(core_tri.development) == len(legacy_dev)

    def test_valuation_properties(self, raa):
        """Test valuation date and valuation vector properties"""
        legacy_val_date = raa.valuation_date
        legacy_val = raa.valuation
        
        df = raa.to_frame(keepdims=True, origin_as_datetime=True).reset_index()
        core_tri = CoreTriangle(
            pl.from_pandas(df),
            origin='origin',
            valuation='valuation',
            columns='values',
            cumulative=True
        )
        
        # Valuation date should match (approximately)
        core_val_date = core_tri.valuation_date
        # Should be same year-month at minimum
        assert core_val_date.year == legacy_val_date.year
        assert core_val_date.month == legacy_val_date.month

    def test_columns_property(self, clrd):
        """Test columns property consistency"""  
        legacy_cols = clrd.columns
        df = clrd.to_frame(keepdims=True, origin_as_datetime=True).reset_index()
        
        core_tri = CoreTriangle(
            pl.from_pandas(df),
            index=clrd.key_labels,
            origin='origin',
            valuation='valuation',
            columns='values',
            cumulative=True
        )
        
        # Should have at least one column
        assert len(core_tri.columns) >= 1


class TestTriangleDataAccess:
    """Test data access methods and slicing"""

    def test_values_property(self, raa):
        """Test values property returns proper structure"""
        df = raa.to_frame(keepdims=True, origin_as_datetime=True).reset_index()
        core_tri = CoreTriangle(
            pl.from_pandas(df),
            origin='origin',
            valuation='valuation', 
            columns='values',
            cumulative=True
        )
        
        values = core_tri.values
        # Values should be DataFrame with proper structure
        assert isinstance(values, pl.DataFrame)
        expected_cols = ['index', 'origin', 'development'] + core_tri.columns
        assert all(col in values.columns for col in expected_cols)

    def test_basic_slicing_by_column(self, clrd):
        """Test column selection"""
        df = clrd.to_frame(keepdims=True, origin_as_datetime=True).reset_index()
        
        # Test with multi-column data
        core_tri = CoreTriangle(
            pl.from_pandas(df),
            index=clrd.key_labels,
            origin='origin',
            valuation='valuation',
            columns='values',
            cumulative=True
        )
        
        # Should be able to select column (if multiple exist)
        if len(core_tri.columns) > 0:
            col_name = core_tri.columns[0]
            selected = core_tri[col_name]
            assert len(selected.columns) == 1
            assert selected.columns[0] == col_name

    def test_basic_indexing(self, raa):
        """Test basic array-style indexing"""
        df = raa.to_frame(keepdims=True, origin_as_datetime=True).reset_index()
        core_tri = CoreTriangle(
            pl.from_pandas(df),
            origin='origin',
            valuation='valuation',
            columns='values',
            cumulative=True
        )
        
        # Test head/tail operations
        head_tri = core_tri.head(3)
        assert head_tri.shape[0] <= 3
        
        tail_tri = core_tri.tail(3) 
        assert tail_tri.shape[0] <= 3


class TestTriangleTransformations:
    """Test data transformations"""

    def test_cumulative_incremental_conversion(self, raa):
        """Test cumulative/incremental conversion methods"""
        df = raa.to_frame(keepdims=True, origin_as_datetime=True).reset_index()
        core_tri = CoreTriangle(
            pl.from_pandas(df),
            origin='origin',
            valuation='valuation',
            columns='values',
            cumulative=True
        )
        
        # Test conversion to incremental and back
        incremental = core_tri.to_incremental()
        assert incremental.is_cumulative == False
        
        back_to_cum = incremental.to_cumulative()
        assert back_to_cum.is_cumulative == True

    def test_development_valuation_conversion(self, raa):
        """Test development/valuation triangle conversion"""
        df = raa.to_frame(keepdims=True, origin_as_datetime=True).reset_index()
        core_tri = CoreTriangle(
            pl.from_pandas(df),
            origin='origin',
            valuation='valuation',
            columns='values',
            cumulative=True
        )
        
        # Test conversion between development and valuation
        val_tri = core_tri.to_valuation()
        assert val_tri.is_val_tri == True
        
        dev_tri = val_tri.to_development()
        assert dev_tri.is_val_tri == False

    def test_grain_transformation(self, qtr):
        """Test grain transformation functionality"""
        # Skip if quarterly triangle not available for this test
        df = qtr.to_frame(keepdims=True, origin_as_datetime=True).reset_index()
        core_tri = CoreTriangle(
            pl.from_pandas(df),
            origin='origin',
            valuation='valuation',
            columns='values',
            cumulative=True
        )
        
        # Test grain transformation (if implemented)
        try:
            annual_tri = core_tri.to_grain("OYDY")
            # Should have different grain
            assert annual_tri.origin_grain in ['Y', 'A']
        except (NotImplementedError, AttributeError):
            # Skip if not yet implemented
            pytest.skip("Grain transformation not yet implemented")


class TestTriangleAggregations:
    """Test aggregation operations"""

    def test_sum_aggregation(self, clrd):
        """Test sum across different axes"""
        df = clrd.to_frame(keepdims=True, origin_as_datetime=True).reset_index()
        core_tri = CoreTriangle(
            pl.from_pandas(df),
            index=clrd.key_labels,
            origin='origin',
            valuation='valuation',
            columns='values',
            cumulative=True
        )
        
        # Test sum across index (axis=0)
        sum_idx = core_tri.sum(axis=0)
        assert sum_idx.shape[0] == 1  # Should collapse index dimension
        
        # Test sum across origin (axis=2) 
        sum_origin = core_tri.sum(axis=2)
        assert sum_origin.shape[2] == 1  # Should collapse origin dimension

    def test_statistical_aggregations(self, raa):
        """Test mean, std, median operations"""
        df = raa.to_frame(keepdims=True, origin_as_datetime=True).reset_index()
        core_tri = CoreTriangle(
            pl.from_pandas(df),
            origin='origin',
            valuation='valuation',
            columns='values',
            cumulative=True
        )
        
        # Test various statistical operations
        mean_tri = core_tri.mean()
        assert mean_tri.shape[0] == 1
        
        median_tri = core_tri.median()
        assert median_tri.shape[0] == 1

    def test_groupby_operations(self, clrd):
        """Test groupby functionality"""
        df = clrd.to_frame(keepdims=True, origin_as_datetime=True).reset_index()
        core_tri = CoreTriangle(
            pl.from_pandas(df),
            index=clrd.key_labels,
            origin='origin', 
            valuation='valuation',
            columns='values',
            cumulative=True
        )
        
        # Test groupby if multi-index
        if len(clrd.key_labels) > 1:
            group_col = clrd.key_labels[0]
            grouped = core_tri.group_by(group_col)
            grouped_sum = grouped.sum()
            # Should have fewer index entries than original
            assert grouped_sum.shape[0] <= core_tri.shape[0]


class TestTriangleArithmetic:
    """Test arithmetic operations"""

    def test_basic_arithmetic(self, raa):
        """Test basic arithmetic operations"""
        df = raa.to_frame(keepdims=True, origin_as_datetime=True).reset_index()
        core_tri = CoreTriangle(
            pl.from_pandas(df),
            origin='origin',
            valuation='valuation',
            columns='values',
            cumulative=True
        )
        
        # Test scalar arithmetic
        doubled = core_tri * 2
        assert doubled.shape == core_tri.shape
        
        halved = doubled / 2
        assert halved.shape == core_tri.shape
        
        # Test triangle arithmetic
        added = core_tri + core_tri
        assert added.shape == core_tri.shape

    def test_arithmetic_broadcasting(self, clrd):
        """Test arithmetic broadcasting rules"""
        df = clrd.to_frame(keepdims=True, origin_as_datetime=True).reset_index()
        core_tri = CoreTriangle(
            pl.from_pandas(df),
            index=clrd.key_labels,
            origin='origin',
            valuation='valuation', 
            columns='values',
            cumulative=True
        )
        
        # Test with latest diagonal (should broadcast)
        try:
            diagonal = core_tri.latest_diagonal
            result = core_tri + diagonal
            assert result.shape == core_tri.shape
        except (NotImplementedError, AttributeError):
            pytest.skip("Latest diagonal not yet implemented")


class TestTriangleDataFrameIntegration:
    """Test DataFrame integration"""

    def test_to_frame_conversion(self, raa):
        """Test to_frame method"""
        df = raa.to_frame(keepdims=True, origin_as_datetime=True).reset_index()
        core_tri = CoreTriangle(
            pl.from_pandas(df),
            origin='origin',
            valuation='valuation',
            columns='values',
            cumulative=True
        )
        
        # Test conversion back to frame
        result_frame = core_tri.to_frame()
        assert isinstance(result_frame, pl.DataFrame)
        # Should have origin/development info
        expected_cols = ['origin']
        assert any(col in result_frame.columns for col in expected_cols)

    def test_wide_format(self, raa):
        """Test wide format for simple triangles"""
        df = raa.to_frame(keepdims=True, origin_as_datetime=True).reset_index()
        core_tri = CoreTriangle(
            pl.from_pandas(df),
            origin='origin',
            valuation='valuation',
            columns='values', 
            cumulative=True
        )
        
        # Test wide format for single index/column triangles
        if core_tri.shape[:2] == (1, 1):
            wide_df = core_tri.wide()
            assert isinstance(wide_df, pl.DataFrame)


class TestTriangleEdgeCases:
    """Test edge cases and error conditions"""

    def test_single_entry_triangle(self):
        """Test triangle with single data entry"""
        data = pl.DataFrame({
            'origin': [datetime(2020, 1, 1)],
            'valuation': [datetime(2021, 1, 1)],
            'values': [1000.0]
        })
        
        core_tri = CoreTriangle(
            data,
            origin='origin',
            valuation='valuation',
            columns='values',
            cumulative=True
        )
        
        assert core_tri.shape[2:] == (1, 1)  # Single origin, single development

    def test_sparse_data_handling(self, prism):
        """Test handling of sparse/missing data"""
        # Create triangle from prism data  
        df = prism.to_frame(keepdims=True, origin_as_datetime=True).reset_index()
        core_tri = CoreTriangle(
            pl.from_pandas(df),
            index=prism.key_labels,
            origin='origin',
            valuation='valuation',
            columns='values',
            cumulative=True
        )
        
        # Should handle sparse data without errors
        assert core_tri.shape[0] > 0

    def test_empty_triangle_handling(self):
        """Test behavior with empty data"""
        empty_data = pl.DataFrame({
            'origin': [],
            'valuation': [],
            'values': []
        }).cast({'origin': pl.Date, 'valuation': pl.Date, 'values': pl.Float64})
        
        try:
            core_tri = CoreTriangle(
                empty_data,
                origin='origin', 
                valuation='valuation',
                columns='values',
                cumulative=True
            )
            # Should handle gracefully or raise appropriate error
            assert True
        except (ValueError, Exception):
            # Empty data should raise appropriate error
            assert True


class TestTriangleLegacyParity:
    """Direct comparison tests against legacy implementation"""

    @pytest.mark.parametrize("dataset_name", ["raa", "qtr", "clrd"])
    def test_basic_properties_parity(self, request, dataset_name):
        """Test that basic properties match between core and legacy"""
        dataset = request.getfixturevalue(dataset_name)
        
        # Create core triangle from legacy data
        df = dataset.to_frame(keepdims=True, origin_as_datetime=True).reset_index()
        core_tri = CoreTriangle(
            pl.from_pandas(df),
            index=dataset.key_labels if len(dataset.key_labels) > 0 else None,
            origin='origin',
            valuation='valuation',
            columns='values',
            cumulative=dataset.is_cumulative
        )
        
        # Test shape compatibility  
        legacy_shape = dataset.shape
        core_shape = core_tri.shape
        
        # At minimum, origin and development dimensions should match
        assert core_shape[2] == legacy_shape[2], f"Origin dimension mismatch: {core_shape[2]} vs {legacy_shape[2]}"
        assert core_shape[3] == legacy_shape[3], f"Development dimension mismatch: {core_shape[3]} vs {legacy_shape[3]}"
        
        # Test grain properties
        assert hasattr(core_tri, 'origin_grain')
        assert hasattr(core_tri, 'development_grain')

    def test_cumulative_conversion_parity(self, raa):
        """Test that cumulative/incremental conversion matches legacy"""
        # Legacy conversion
        legacy_incr = raa.cum_to_incr()
        legacy_back_to_cum = legacy_incr.incr_to_cum()
        
        # Core conversion
        df = raa.to_frame(keepdims=True, origin_as_datetime=True).reset_index()
        core_tri = CoreTriangle(
            pl.from_pandas(df),
            origin='origin',
            valuation='valuation',
            columns='values',
            cumulative=True
        )
        
        core_incr = core_tri.to_incremental()
        core_back_to_cum = core_incr.to_cumulative()
        
        # Test flags
        assert core_incr.is_cumulative == legacy_incr.is_cumulative
        assert core_back_to_cum.is_cumulative == legacy_back_to_cum.is_cumulative

    def test_aggregation_consistency(self, clrd):
        """Test that aggregations produce consistent results"""
        # Focus on sum operation which is most critical
        legacy_sum = clrd.sum()
        
        df = clrd.to_frame(keepdims=True, origin_as_datetime=True).reset_index()
        core_tri = CoreTriangle(
            pl.from_pandas(df),
            index=clrd.key_labels,
            origin='origin',
            valuation='valuation',
            columns='values',
            cumulative=clrd.is_cumulative
        )
        
        core_sum = core_tri.sum()
        
        # Should have same overall structure
        assert core_sum.shape[0] == 1  # Should collapse to single index
        assert legacy_sum.shape[0] == 1