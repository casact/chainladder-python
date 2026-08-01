# {octicon}`book` User Guide

Here are a few examples that we have came up with, hopefully the code is simple enough so you can fit your own needs.

`````{grid}
:gutter: 3

````{grid-item-card}
:columns: 6

**{doc}`Triangles <triangle>`**
^^^

Data object to manage and manipulate reserving data

**Application**: Extend pandas syntax to manipulate reserving triangles

```{image} ../images/plot_triangle_from_pandas.png
```
+++
**Classes**: {ref}`Triangle <triangle:triangle>`, …
````

````{grid-item-card}
:columns: 6

**{doc}`Development <development>`**
^^^
Tooling to generate loss development patterns

**Applications**: Comprehensive library for development

```{image} ../images/plot_clarkldf.png
```
+++

**Algorithms**: {ref}`Development <development:development>`,
{ref}`ClarkLDF <development:clarkldf>`, …

````

````{grid-item-card}
:columns: 6

**{doc}`Tail Estimation <tails>`**
^^^
Extrapolate development patterns beyond the known data.

**Applications**: Long-tailed lines of business use cases

```{image} ../images/plot_exponential_smoothing.png
```

+++
**Algorithms**: {ref}`TailCurve <tails:tailcurve>`,
{ref}`TailConstant <tails:tailconstant>`, …
````

````{grid-item-card}
:columns: 6

**{doc}`IBNR Models <methods>`**
^^^

Generate IBNR estimates and associated statistics


**Applications**: Constructing reserve estimates

```{image} ../images/plot_mack.png
```

+++
**Algorithms**: {ref}`Chainladder <methods:chainladder>`,
{ref}`CapeCod <methods:capecod>`, …
````

````{grid-item-card}
:columns: 6

**{doc}`Adjustments <adjustments>`**
^^^
Common actuarial data adjustments



**Applications**: Simulation, trending, on-leveling

```{image} ../images/plot_stochastic_bornferg.png
```

+++
**Classes**: {ref}`BootstrapODPSample <adjustments:bootstrapodpsample>`,
{ref}`Trend <adjustments:trend>`, …
````

````{grid-item-card}
:columns: 6

**{doc}`Workflow <workflow>`**
^^^

Workflow tools for complex analyses

**Application**: Scenario testing, ensembling

```{image} ../images/plot_voting_chainladder.png
```

+++
**Utilities**: {ref}`Pipeline <workflow:pipeline>`,
{ref}`VotingChainladder <workflow:votingchainladder>`, …
````

`````
