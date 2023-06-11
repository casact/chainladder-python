# {octicon}`book` User Guide

Here are a few examples that we have came up with, hopefully the code is simple enough so you can fit your own needs.

`````{grid}
:gutter: 3

````{grid-item-card}
:columns: 6

**[Triangles](triangle)**
^^^

Data object to manage and manipulate reserving data

**Application**: Extend pandas syntax to manipulate reserving triangles

```{glue:} plot_triangle_from_pandas
```
+++
**Classes**: **[Triangle](triangle)**, ...
````

````{grid-item-card}
:columns: 6

**[Development](development)**
^^^
Tooling to generate loss development patterns

**Applications**: Comprehensive library for development

```{glue:} plot_clarkldf
```
+++

**Algorithms**: [Development](development:development), [ClarkLDF](development:clarkldf), …

````

````{grid-item-card}
:columns: 6

**[Tail Estimation](tails)**
^^^
Extrapolate development patterns beyond the known data.

**Applications**: Long-tailed lines of business use cases

```{glue:} plot_exponential_smoothing
```

+++
**Algorithms**: [TailCurve](tails:tailcurve), [TailConstant](tails:tailconstant), …
````

````{grid-item-card}
:columns: 6

**[IBNR Models](methods)**
^^^

Generate IBNR estimates and associated statistics


**Applications**: Constructing reserve estimates

```{glue:} plot_mack
```

+++
**Algorithms**: [Chainladder](methods:chainladder), [CapeCod](methods:capecod), …
````

````{grid-item-card}
:columns: 6

**[Adjustments](adjustments)**
^^^
Common actuarial data adjustments



**Applications**: Simulation, trending, on-leveling

```{glue:} plot_stochastic_bornferg
```

+++
**Classes**: [BootstrapODPSample](adjustments:bootstrapodpsample), [Trend](adjustments:trend), …
````

````{grid-item-card}
:columns: 6

**[Workflow](workflow)**
^^^

Workflow tools for complex analyses

**Application**: Scenario testing, ensembling

```{glue:} plot_voting_chainladder
```

+++
**Utilities**: [Pipeline](workflow:pipeline), [VotingChainladder](workflow:votingchainladder), …
````

`````
