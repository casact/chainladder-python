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
**Classes**: **{py:class}`chainladder.Triangle`**, ...
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

**Algorithms**: {py:class}`chainladder.Development`,
{py:class}`chainladder.ClarkLDF`, …

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
**Algorithms**: {py:class}`chainladder.TailCurve`,
{py:class}`chainladder.TailConstant`, …
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
**Algorithms**: {py:class}`chainladder.Chainladder`,
{py:class}`chainladder.CapeCod`, …
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
**Classes**: {py:class}`chainladder.BootstrapODPSample`,
{py:class}`chainladder.Trend`, …
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
**Utilities**: {py:class}`chainladder.Pipeline`,
{py:class}`chainladder.VotingChainladder`, …
````

`````
