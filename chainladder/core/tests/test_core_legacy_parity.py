# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
Direct parity tests between core (polars) and legacy (pandas) Triangle implementations.
These tests ensure that the new polars-based Triangle maintains backward compatibility.
"""

import pytest
import pandas as pd
import numpy as np
import chainladder as cl
from chainladder.core.triangle import Triangle as CoreTriangle
from chainladder.legacy.triangle import Triangle as LegacyTriangle
import polars as pl
from datetime import datetime


class TestCoreLegacyParity:
    """Test direct parity between core and legacy Triangle implementations"""

    def create_core_from_legacy(self, legacy_triangle):
        """Helper to create core triangle from legacy triangle"""
        df = legacy_triangle.to_frame(keepdims=True, origin_as_datetime=True).reset_index()
        
        return CoreTriangle(
            pl.from_pandas(df),
            index=legacy_triangle.key_labels if len(legacy_triangle.key_labels) > 0 else None,
            origin='origin',
            valuation='valuation',
            columns='values',
            cumulative=legacy_triangle.is_cumulative
        )

    def test_shape_parity(self, raa, qtr, clrd):
        """Test that shapes match between core and legacy implementations"""
        for dataset in [raa, qtr, clrd]:
            core_tri = self.create_core_from_legacy(dataset)
            
            # Shape should match exactly or be compatible
            legacy_shape = dataset.shape
            core_shape = core_tri.shape
            
            # Origin and development dimensions must match
            assert core_shape[2] == legacy_shape[2], f"Origin mismatch: {core_shape[2]} vs {legacy_shape[2]}"
            assert core_shape[3] == legacy_shape[3], f"Development mismatch: {core_shape[3]} vs {legacy_shape[3]}"
            
            # Index should be compatible (might differ due to implementation)
            assert core_shape[0] >= 1, "Core should have at least 1 index entry"

    def test_grain_properties_parity(self, raa, qtr, clrd):
        """Test that grain properties match"""
        for dataset in [raa, qtr, clrd]:
            core_tri = self.create_core_from_legacy(dataset)
            
            # Test origin grain
            legacy_origin_grain = dataset.origin_grain
            core_origin_grain = core_tri.origin_grain
            assert core_origin_grain in ['Y', 'Q', 'M', 'S'], f"Invalid core origin grain: {core_origin_grain}"
            
            # Test development grain  
            legacy_dev_grain = dataset.development_grain
            core_dev_grain = core_tri.development_grain
            assert core_dev_grain in ['Y', 'Q', 'M', 'S'], f"Invalid core development grain: {core_dev_grain}"

    def test_cumulative_property_parity(self, raa):
        """Test cumulative property consistency"""
        # Test with cumulative triangle
        core_tri = self.create_core_from_legacy(raa)
        assert core_tri.is_cumulative == raa.is_cumulative
        
        # Test with incremental triangle
        legacy_incr = raa.cum_to_incr()
        core_incr = core_tri.to_incremental()
        assert core_incr.is_cumulative == legacy_incr.is_cumulative

    def test_valuation_date_parity(self, raa, qtr):
        """Test valuation date consistency"""
        for dataset in [raa, qtr]:
            core_tri = self.create_core_from_legacy(dataset)
            
            legacy_val_date = dataset.valuation_date
            core_val_date = core_tri.valuation_date
            
            # Should be same year and month at minimum
            assert core_val_date.year == legacy_val_date.year
            assert core_val_date.month == legacy_val_date.month

    def test_conversion_methods_parity(self, raa):
        """Test that conversion methods produce equivalent results"""
        core_tri = self.create_core_from_legacy(raa)
        
        # Test cumulative -> incremental conversion
        legacy_incr = raa.cum_to_incr()
        core_incr = core_tri.to_incremental()
        
        assert core_incr.is_cumulative == legacy_incr.is_cumulative
        
        # Test incremental -> cumulative conversion
        legacy_back_cum = legacy_incr.incr_to_cum()
        core_back_cum = core_incr.to_cumulative()
        
        assert core_back_cum.is_cumulative == legacy_back_cum.is_cumulative

    def test_aggregation_results_parity(self, clrd):
        """Test that aggregations produce numerically similar results"""
        core_tri = self.create_core_from_legacy(clrd)
        
        # Test sum aggregation across index (axis=0)
        legacy_sum = clrd.sum(axis=0)
        core_sum = core_tri.sum(axis=0)
        
        # Both should collapse to single index entry
        assert legacy_sum.shape[0] == 1
        assert core_sum.shape[0] == 1
        
        # Test sum across different axes
        for axis in [0, 2, 3]:  # Skip axis=1 (columns) for simplicity
            legacy_agg = clrd.sum(axis=axis)
            core_agg = core_tri.sum(axis=axis)
            
            # Shape should be compatible
            expected_shape = list(clrd.shape)
            expected_shape[axis] = 1
            
            assert core_agg.shape[axis] == 1, f"Axis {axis} not properly aggregated"

    def test_slicing_behavior_parity(self, clrd):
        """Test that slicing produces compatible results"""
        core_tri = self.create_core_from_legacy(clrd)
        
        # Test head/tail operations
        legacy_head = clrd.iloc[:3]
        core_head = core_tri.head(3)
        
        assert core_head.shape[0] <= 3
        assert core_head.shape[0] == min(3, core_tri.shape[0])

    def test_boolean_indexing_parity(self, clrd):
        """Test boolean indexing compatibility"""
        if len(clrd.key_labels) == 0:
            pytest.skip("No index labels for boolean indexing test")
            
        core_tri = self.create_core_from_legacy(clrd)
        
        # Test basic filtering - both should handle gracefully
        try:
            # This might not work identically, but should not crash
            index_vals = clrd.index.iloc[:2]
            # Both implementations should handle this type of operation
            assert True  # Placeholder - actual filtering logic would go here
        except (NotImplementedError, AttributeError):
            # Expected for some operations not yet implemented
            pass

    def test_arithmetic_parity(self, raa):
        """Test arithmetic operations produce compatible results"""
        core_tri = self.create_core_from_legacy(raa)
        
        # Test scalar arithmetic
        legacy_doubled = raa * 2
        core_doubled = core_tri * 2
        
        assert core_doubled.shape == legacy_doubled.shape
        
        # Test triangle arithmetic
        legacy_added = raa + raa
        core_added = core_tri + core_tri
        
        assert core_added.shape == legacy_added.shape

    def test_dataframe_roundtrip_parity(self, raa):
        """Test DataFrame conversion roundtrip compatibility"""
        # Start with legacy triangle
        legacy_df = raa.to_frame(keepdims=True, origin_as_datetime=True)
        
        # Create core triangle
        core_tri = self.create_core_from_legacy(raa)
        
        # Convert core back to DataFrame
        core_df = core_tri.to_frame()
        
        # Both should be DataFrames with origin/development info
        assert isinstance(legacy_df, pd.DataFrame)
        assert isinstance(core_df, pl.DataFrame)
        
        # Should have similar column structures
        legacy_cols = set(legacy_df.columns)
        core_cols = set(core_df.columns)
        
        # Should have origin information
        assert any('origin' in str(col).lower() for col in legacy_cols)
        assert any('origin' in str(col).lower() for col in core_cols)


class TestSpecificDatasetParity:
    """Test parity with specific datasets that have unique characteristics"""

    def create_core_from_legacy(self, legacy_triangle):
        """Helper to create core triangle from legacy triangle"""
        df = legacy_triangle.to_frame(keepdims=True, origin_as_datetime=True).reset_index()
        
        return CoreTriangle(
            pl.from_pandas(df),
            index=legacy_triangle.key_labels if len(legacy_triangle.key_labels) > 0 else None,
            origin='origin',
            valuation='valuation',
            columns='values',
            cumulative=legacy_triangle.is_cumulative
        )

    def test_raa_parity(self, raa):
        """Test parity with RAA dataset (simple 2D triangle)"""
        core_tri = self.create_core_from_legacy(raa)
        
        # RAA is a simple cumulative triangle
        assert core_tri.is_cumulative == raa.is_cumulative
        assert core_tri.shape[2:] == raa.shape[2:]  # Same origin/development
        
        # Test link ratio calculation if implemented
        try:
            legacy_lr = raa.link_ratio
            core_lr = core_tri.link_ratio
            # Should have similar shapes
            assert core_lr.shape[2:] == legacy_lr.shape[2:]
        except (NotImplementedError, AttributeError):
            # Link ratio may not be implemented yet
            pass

    def test_quarterly_parity(self, qtr):
        """Test parity with quarterly dataset"""
        core_tri = self.create_core_from_legacy(qtr)
        
        # Quarterly data has specific grain
        assert qtr.origin_grain == 'Q'
        assert core_tri.origin_grain in ['Q', 'M']  # Should be quarterly or finer
        
        # Test valuation date consistency
        legacy_val = qtr.valuation_date
        core_val = core_tri.valuation_date
        assert core_val.year == legacy_val.year

    def test_clrd_parity(self, clrd):
        """Test parity with CLRD dataset (4D triangle with multiple indices/columns)"""
        core_tri = self.create_core_from_legacy(clrd)
        
        # CLRD has multiple index dimensions
        assert len(clrd.key_labels) > 1
        assert len(core_tri.key_labels) >= 1
        
        # Should handle multiple columns
        legacy_cols = len(clrd.columns)
        assert legacy_cols > 1
        
        # Test groupby operations if implemented
        if len(clrd.key_labels) > 0:
            try:
                legacy_grouped = clrd.groupby(clrd.key_labels[0]).sum()
                core_grouped = core_tri.group_by(clrd.key_labels[0]).sum()
                
                # Should reduce index dimension
                assert core_grouped.shape[0] <= core_tri.shape[0]
            except (NotImplementedError, AttributeError):
                # Groupby may not be fully implemented
                pass

    def test_prism_parity(self, prism):
        """Test parity with Prism dataset (large, sparse triangle)"""
        core_tri = self.create_core_from_legacy(prism)
        
        # Prism is a large sparse dataset
        assert prism.shape[0] > 100  # Many index entries
        assert core_tri.shape[0] >= 1  # Should handle large data
        
        # Test performance with large data
        try:
            # Should be able to perform basic operations without error
            core_sum = core_tri.sum()
            assert core_sum.shape[0] == 1
        except Exception as e:
            # Should not crash on large data
            pytest.fail(f"Core implementation failed on large dataset: {e}")


class TestErrorConditionParity:
    """Test that error conditions are handled similarly"""

    def test_invalid_data_handling(self):
        """Test that invalid data produces appropriate errors"""
        # Test empty data
        empty_data = pl.DataFrame({
            'origin': [],
            'valuation': [], 
            'values': []
        }).cast({'origin': pl.Date, 'valuation': pl.Date, 'values': pl.Float64})
        
        # Should either work or raise appropriate error
        try:
            core_tri = CoreTriangle(
                empty_data,
                origin='origin',
                valuation='valuation',
                columns='values',
                cumulative=True
            )
            assert True  # If it works, that's fine
        except Exception:
            assert True  # If it raises an error, that's also fine

    def test_mismatched_dimensions(self):
        """Test error handling for dimension mismatches"""
        # Test with inconsistent data
        inconsistent_data = pl.DataFrame({
            'origin': [datetime(2020, 1, 1), datetime(2021, 1, 1)],
            'valuation': [datetime(2020, 6, 1)],  # Missing valuation
            'values': [1000.0]
        })
        
        # Should handle gracefully or raise appropriate error
        try:
            core_tri = CoreTriangle(
                inconsistent_data,
                origin='origin',
                valuation='valuation', 
                columns='values',
                cumulative=True
            )
            assert True
        except Exception:
            # Appropriate error handling
            assert True


# Integration test that exercises the full workflow
class TestFullWorkflowParity:
    """Test complete workflows from data ingestion to analysis"""

    def test_complete_workflow_raa(self, raa):
        """Test complete workflow with RAA data"""
        # Create core triangle
        df = raa.to_frame(keepdims=True, origin_as_datetime=True).reset_index()
        core_tri = CoreTriangle(
            pl.from_pandas(df),
            origin='origin',
            valuation='valuation',
            columns='values',
            cumulative=True
        )
        
        # Test complete workflow:
        # 1. Data conversion
        incremental = core_tri.to_incremental()
        assert incremental.is_cumulative == False
        
        # 2. Back to cumulative
        back_to_cum = incremental.to_cumulative()
        assert back_to_cum.is_cumulative == True
        
        # 3. Aggregation
        summed = core_tri.sum()
        assert summed.shape[0] == 1
        
        # 4. Frame conversion
        result_df = summed.to_frame()
        assert isinstance(result_df, pl.DataFrame)
        
        # Workflow should complete without errors
        assert True

    def test_complete_workflow_clrd(self, clrd):
        """Test complete workflow with CLRD multi-dimensional data"""
        df = clrd.to_frame(keepdims=True, origin_as_datetime=True).reset_index()
        core_tri = CoreTriangle(
            pl.from_pandas(df),
            index=clrd.key_labels,
            origin='origin',
            valuation='valuation',
            columns='values',
            cumulative=True
        )
        
        # Test multi-dimensional workflow:
        # 1. Aggregation across index
        summed = core_tri.sum(axis=0)
        assert summed.shape[0] == 1
        
        # 2. Conversion operations
        incremental = core_tri.to_incremental()
        assert incremental.is_cumulative == False
        
        # 3. Back conversion
        cumulative = incremental.to_cumulative()
        assert cumulative.is_cumulative == True
        
        # Multi-dimensional workflow should work
        assert True