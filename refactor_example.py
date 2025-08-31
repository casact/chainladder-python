#!/usr/bin/env python3
"""
Example of refactoring chainladder models from numpy-based internal API to polars-native public API.

This demonstrates the polars-first, arrow-compatible approach outlined in CLAUDE.md.
"""

import chainladder as cl
import polars as pl


def demonstrate_refactoring():
    """Show before/after refactoring patterns for common operations."""
    
    print("=== Polars-First Refactoring Examples ===\n")
    
    # Load sample data
    raa = cl.load_sample('raa')
    print(f"Working with RAA triangle, shape: {raa.shape}")
    
    print("\n1. REPLACING NON-FINITE VALUES")
    print("=" * 40)
    
    print("OLD numpy-based approach (from chainladder/methods/base.py:48):")
    print("    xp = ultimate.get_array_module()")
    print("    if ultimate.array_backend != 'sparse':")
    print("        ultimate.values[~xp.isfinite(ultimate.values)] = xp.nan")
    print("")
    
    print("NEW polars-native approach:")
    print("    ultimate = ultimate.replace_non_finite()")
    
    # Demonstrate the new approach
    clean_raa = raa.replace_non_finite()
    print(f"✓ Executed replace_non_finite(), shape preserved: {clean_raa.shape}")
    
    print("\n2. MATHEMATICAL OPERATIONS ON TRIANGLE VALUES")  
    print("=" * 50)
    
    print("OLD numpy-based approach (from chainladder/methods/mack.py):")
    print("    obj.values = X.sigma_.values / num_to_nan(weight)")
    print("    obj.values = xp.nan_to_num(obj.values) * xp.array(w)")
    print("")
    
    print("NEW polars-native approach:")
    print("    obj = triangle.with_values(")
    print("        values=pl.col('sigma') / pl.col('weight').fill_null(1)")
    print("    )")
    print("    obj = obj.with_values(")
    print("        values=pl.col('values').fill_nan(0) * pl.col('weight')")  
    print("    )")
    
    # Demonstrate mathematical operations
    # Example: multiply all values by 1.5 (representing some calculation)
    modified_raa = raa.with_values(values=pl.col('values') * 1.5)
    
    # Verify the operation worked
    orig_sum = raa.values.select('0').sum().item()
    new_sum = modified_raa.values.select('0').sum().item()
    expected_sum = orig_sum * 1.5
    
    if abs(new_sum - expected_sum) < 0.01:
        print(f"✓ Mathematical operation successful: {orig_sum:.0f} → {new_sum:.0f}")
    else:
        print("✗ Mathematical operation failed")
    
    print("\n3. ARRAY CONCATENATION AND RESHAPING")
    print("=" * 40)
    
    print("OLD numpy-based approach (from chainladder/methods/mack.py):")
    print("    cols = (self.latest_diagonal.values, self.ibnr_.values, ...)")
    print("    obj.values = obj.get_array_module().concatenate(cols, 3)")
    print("")
    
    print("NEW polars-native approach:")
    print("    # Use polars horizontal concatenation")
    print("    combined = pl.concat([")
    print("        triangle1.values.select(['index', 'origin', 'development', '0']),")
    print("        triangle2.values.select('0').rename({'0': '1'}),")
    print("        triangle3.values.select('0').rename({'0': '2'})")
    print("    ], how='horizontal')")
    
    # Demonstrate concatenation by creating multiple derived triangles
    latest_diag = raa.latest_diagonal
    link_ratios = raa.link_ratio
    
    print(f"✓ Created derived triangles:")
    print(f"  - Latest diagonal: {latest_diag.shape}")  
    print(f"  - Link ratios: {link_ratios.shape}")
    
    print("\n4. STATISTICAL OPERATIONS")
    print("=" * 25)
    
    print("OLD approach: Direct numpy array operations")
    print("NEW approach: Use polars expressions for statistics")
    
    # Demonstrate statistical operations using polars
    stats_result = raa.values.select([
        pl.col('0').mean().alias('mean_value'),
        pl.col('0').std().alias('std_value'), 
        pl.col('0').max().alias('max_value'),
        pl.col('0').min().alias('min_value'),
        pl.col('0').count().alias('count_non_null')
    ])
    
    print("✓ Statistical summary using polars expressions:")
    print(stats_result.to_pandas().to_string(index=False))
    
    print("\n=== KEY BENEFITS OF POLARS-FIRST APPROACH ===")
    print("✓ Arrow compatibility - seamless interoperability")
    print("✓ Better performance - lazy evaluation and query optimization") 
    print("✓ Type safety - compile-time column validation")
    print("✓ Expressiveness - declarative data transformations")
    print("✓ Consistency - unified API across all operations")
    
    print("\n=== REFACTORING STRATEGY ===")
    print("1. Replace direct .values assignments with .with_values()")
    print("2. Replace numpy array operations with polars expressions")
    print("3. Use .replace_non_finite() instead of manual NaN handling")
    print("4. Leverage polars' built-in statistical and mathematical functions")
    print("5. Use polars concat instead of numpy concatenate")
    
    return clean_raa, modified_raa


if __name__ == "__main__":
    demonstrate_refactoring()