# WELCOME!

`chainladder-python` - Property and Casualty Loss Reserving in Python

Welcome! The `chainladder-python` package was built to be able to handle all of your actuarial needs in python. It consists of popular actuarial tools, such as triangle data manipulation, link ratios calculation, and IBNR estimates using both deterministic and stochastic models. We build this package so you no longer have to rely on outdated softwares and tools when performing actuarial pricing or reserving indications.

This package strives to be minimalistic in needing its own API. The syntax mimics popular packages such as [pandas](https://pandas.pydata.org/) for data manipulation and [scikit-learn](https://scikit-learn.org/) for model construction. An actuary that is already familiar with these tools will be able to pick up this package with ease. You will be able to save your mental energy for actual actuarial work.

Don't know where to start? Head over to the `START HERE` section to explore what's next.
=======
Chainladder is built by a group of volunteers, and we need YOUR help!

Here are a few links to help you get started.

::::{grid}
:gutter: 3

:::{grid-item-card} 
:columns: 6
**[Triangles](triangle)**
^^^
Data object to manage and manipulate reserving data

**Application**: Extend pandas syntax to manipulate reserving triangles

```{glue:} plot_triangle_from_pandas
```
+++
**Classes**: **[Triangle](triangle)**,...
:::

:::{grid-item-card} 
:columns: 6

**[Development](development)**
^^^
Tooling to generate loss development patterns

**Applications**: Comprehensive library for development

```{glue:} plot_clarkldf
```

+++
**Algorithms**: [Development](development:development), [ClarkLDF](development:clarkldf), …
:::

:::{grid-item-card}
:columns: 6

**[Tail Estimation](tails)**
^^^
Extrapolate development patterns beyond the known data.

**Applications**: Long-tailed lines of business use cases

```{glue:} plot_exponential_smoothing
```

+++
**Algorithms**: [TailCurve](tails:tailcurve), [TailConstant](tails:tailconstant), …
:::

:::{grid-item-card}
:columns: 6

**[IBNR Models](methods)**
^^^

Generate IBNR estimates and associated statistics


**Applications**: Constructing reserve estimates

```{glue:} plot_mack
```

+++
**Algorithms**: [Chainladder](methods:chainladder), [CapeCod](methods:capecod), …
:::

:::{grid-item-card}
:columns: 6

**[Adjustments](adjustments)**
^^^
Common actuarial data adjustments



**Applications**: Simulation, trending, on-leveling

```{glue:} plot_stochastic_bornferg
```

+++
**Classes**: [BootstrapODPSample](adjustments:bootstrapodpsample), [Trend](adjustments:trend), …
:::

:::{grid-item-card}
:columns: 6

**[Workflow](workflow)**
^^^

Workflow tools for complex analyses

**Application**: Scenario testing, ensembling

```{glue:} plot_voting_chainladder
```

+++
**Utilities**: [Pipeline](workflow:pipeline), [VotingChainladder](workflow:votingchainladder), …
:::

::::
