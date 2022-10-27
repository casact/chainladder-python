# Glossary

This glossary hopes to definitively represent the tacit and explicit
conventions applied in `chanladder` and its API.

## General Terms

backend
  -   The storage of the numerical representation of a
      [Triangle]{.title-ref}. It can be 'numpy' for a dense CPU bound
      representation, 'sparse' for a sparse matrix CPU bound representation,
      or 'cupy' for a dense GPU bound representation.

estimator
  -   Any scikit-learn style class that can be `fit` to Triangle data.

hyperparameter
  -   An initial parameter of an estimator that can be set before the estimator is fit.

predictor
  -   An estimator that has the
  `predict` method. All IBNR estimators of `chainladder` are predictors

transformer
  -   An estimator that has the `transform` method. The transform
      method returns instances of a Triangle. All estimators
      other than IBNR estimators are transformers.

## Class API

Triangle
  -   The core data structure of the `chainladder` package. It emulates
      `pandas`'s functionality.

Development
  -   Transformers that produce, at a minimum, a multiplicative `ldf_`
      property.

Tail
  -   Transformers that extends the Development estimator properties
      beyond the edge of a Triangle.

IBNR
  -   Predictors that produce, at a minimum, an `ultimate_`, `ibnr_`,
      `full_expectation_` and `full_triangle_` property.

Workflow
  -   Meta-estimators that allow for composition of other estimators.

Adjustments
  -   Estimators that allow for the adjustment of the values of a
      Triangle.

## Triangle Concepts

axis
  -   Represents one of the four dimensions of a [Triangle]{.title-ref}
      instance. The four axes are `index`, `columns`, `origin` and
      `development`. `valuation` represents an additional axis implicit in
      the Triangle.

index
  -   The first axis of a 4D Triangle instance. Usually reserved for
      lines of business, or segments.

columns
  -   The second axis of a Triangle.

origin
  -   The third axis of a Triangle that represents origin dates of a Triangle.

development
  -   The fourth axis of a Triangle that represents either development age
      or valuation dates of a Triangle.

valuation
  -   An implicit axis representing the valuation period of each of the
      cells of a `Triangle`.
