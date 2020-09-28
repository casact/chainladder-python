### Core


'/chainladder/core' houses the core data structure of the library, the `Triangle`.

#### File descriptions
**base.py** - Functionality for initializing a triangle.
<br>**slice.py** - Functionality for `loc` and `iloc` slicing, boolean-indexing,
<br>**pandas.py** - Pandas style methods extended to Triangle class such as `groupby` functionality
<br>**io.py** - pickle and json serialization
<br>**triangle.py** - chainladder specific methods not covered by pandas
<br>**dunders.py** - Operator overloading (add, subtract, multiply, divide)
<br>**display.py** - Jupyter and REPL outputs of Triangle.
<br>**common.py** -Common methods that can used by both Triangles and Estimators
<br>**correlation.py** - Mack correlation tests
