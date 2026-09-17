{{ objname | escape | underline }}

.. currentmodule:: {{ module }}

{% set documented_attrs = ['cdf_', 'ibnr_', 'pct_reported_'] %}
{% set hidden_attrs = attributes | reject('in', documented_attrs) | list %}

.. autoclass:: {{ objname }}
   :members:
   :undoc-members:
   :exclude-members: set_fit_request, set_predict_request, set_score_request, set_transform_request{% if hidden_attrs %}, {{ hidden_attrs | join(', ') }}{% endif %}

{% set inherited = [] %}
{% for method in methods %}
{% if method in inherited_members and not method.startswith('_') %}
{% set _ = inherited.append(method) %}
{% endif %}
{% endfor %}

{% if inherited %}
.. rubric:: Inherited Methods

.. autosummary::
   :nosignatures:

{% for method in inherited %}
   {{ objname }}.{{ method }}
{% endfor %}
{% endif %}
