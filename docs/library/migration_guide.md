# Migration Guide for Chainladder 1.0

We will be implementing a number of breaking changes and deprecating others in the upcoming `chainladder-python` 1.0. This page walks you through each item with rationale for the change and guidance on how to address. 

---

## [BRK] Dask and Cupy are removed

`chainladder-python` 1.0 will no longer support Dask and Cupy backends.

We are removing the support for these backend because we have been unable to test new features on these backends. We are also not aware existing users of these backends.

---

## `inplace` in `Triangle.astype` is removed

 `Triangle.astype` will always return a new copy in 1.0. The deprecation warning has been in place since 0.11.0. This is done to mirror Pandas behavior. To prepare for this removal, change your code to the following:

```python
tri = tri.copy()
tri = tri.astype(new_dtype)
```

---

## `tri == other` is renamed to `tri.equals(other)`

In `chainladder-python` 1.0, calling `tri == other` will return a Triangle of `bool`'s. The existing behavior is renamed to `Triangle.equals`. The deprecation warning has been in place since 0.11.0. This is done to mirror Pandas behavior.

---

## [BRK] `PTF_Formula` has been renamed to `ptf_formula`

`PTF_Formula` will be removed in 1.0. The deprecation warning has been in place since 0.11.0. The new name conforms to PEP-8. 

---

## [DEPR] `development` when declaring a `Triangle` is deprecated in favor of `valuation` and `age`

`development_format` is also deprecated for `valuation_format`. The deprecation warning for both has been in place since 0.11. They will be removed in 2.0.

`development` was always a misnomer since the earlier days of this package. The actual values supplied has always been valuation dates. This renaming makes a key API of this package more intuitive.

---

## [DEPR] Sparse will soon be deprecated

`chainladder-python` 2.0 will no longer support Sparse backend. A replacement feature is under discussion and will be announced before version 1.1.

Sparse is time-intensive to maintain and a barrier to ISO-compliant treatment of NaN. Removing Sparse will allow for faster devleopment of new features and methods. We will look to retain key use cases, such as claim-level ultimates.

---

## [DEPR] `inplace=True` returning the mutated object is deprecated

In `chainladder-python` 2.0, calling a method with the `inplace` argument will no longer return the mutated object when `inplace=True`. These methods are affected

- `Triangle.set_backend()`
- `Triangle.set_index()`
- `Triangle.incr_to_cum()`
- `Triangle.cum_to_incr()`
- `Triangle.dev_to_val()`
- `Triangle.val_to_dev()`
- `Triangle.grain()`
- `Triangle.fillna()`
- `Triangle.fillzero()`
- `Triangle.astype()`

A deprecation note has been added to the API reference. This is done to mirror Pandas behavior.

---

## [DEPR] Passing slices to `Triangle.at` is deprecated.

In `chainladder-python` 2.0, neither `raa.at[:, :, "1985", 24]` nor `raa.at[0, 0, "1985", slice(24, 25)]` will work. The deprecation warning has been in place since 0.11.0. This is done to mirror Pandas behavior.
