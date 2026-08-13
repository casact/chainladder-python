{{ objname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :inherited-members:
   :undoc-members:
   :special-members: __add__, __sub__, __mul__, __truediv__
   :exclude-members: set_fit_request, set_predict_request, set_score_request, set_transform_request, {{ attributes | join(', ') }}
