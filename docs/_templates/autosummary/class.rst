{{ objname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :undoc-members:
   :exclude-members: {{ attributes | join(', ') }}

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
