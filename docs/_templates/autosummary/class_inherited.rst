{{ objname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :inherited-members:
   :undoc-members:
   :exclude-members: set_fit_request, set_predict_request, set_score_request, set_transform_request, {{ attributes | join(', ') }}
