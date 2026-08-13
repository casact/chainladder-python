{{ objname | escape | underline }}

.. currentmodule:: {{ module }}

{% set documented_attrs = ['shape', 'empty', 'dimensionality', 'nan_triangle'] %}
{% set hidden_attrs = attributes | reject('in', documented_attrs) | list %}

.. autoclass:: {{ objname }}
   :members:
   :inherited-members:
   :undoc-members:
   :exclude-members: set_fit_request, set_predict_request, set_score_request, set_transform_request{% if hidden_attrs %}, {{ hidden_attrs | join(', ') }}{% endif %}
