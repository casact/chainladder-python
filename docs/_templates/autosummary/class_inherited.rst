{{ objname | escape | underline }}

.. currentmodule:: {{ module }}

{% set indexer_attrs = ['loc', 'iloc', 'at', 'iat'] %}
{% set hidden_attrs = attributes | reject('in', indexer_attrs) | list %}

.. autoclass:: {{ objname }}
   :members:
   :inherited-members:
   :undoc-members:
   :exclude-members: set_fit_request, set_predict_request, set_score_request, set_transform_request{% if hidden_attrs %}, {{ hidden_attrs | join(', ') }}{% endif %}
