# Welcome to Chainladder

`Chainladder-python` - A python library for casualty loss reserving.

> * Simple and efficient tools for actuarial loss reserving
> * Designed with practical workflows in mind
> * Looks like Pandas and Scikit-Learn, the tools you love!
> * Open source, commercially usable - MPL-2.0 license


Here are a few links to help you get started.

:::{panels}
:container: +full-width
:column: col-lg-4 px-2 py-2
---
:header: bg-jb-one

**[Triangles](triangle)**
^^^
Data object to manage and manipulate reserving data

**Application**: Extend pandas syntax to manipulate reserving triangles

```{glue:} plot_triangle_from_pandas
```
+++
**Classes**: **[Triangle](triangle)**,...

---
:header: bg-jb-two

**[Development](development)**
^^^
Tooling to generate loss development patterns

**Applications**: Comprehensive library for development

```{glue:} plot_clarkldf
```

+++
**Algorithms**: [Development](development:development), [ClarkLDF](development:clarkldf), …
---
:header: bg-jb-three

**[Tail Estimation](tails)**
^^^
Extrapolate development patterns beyond the known data.

**Applications**: Long-tailed lines of business use cases

```{glue:} plot_exponential_smoothing
```

+++
**Algorithms**: [TailCurve](tails:tailcurve), [TailConstant](tails:tailconstant), …

---
:header: bg-jb-one
**[IBNR Models](methods)**
^^^

Generate IBNR estimates and associated statistics


**Applications**: Constructing reserve estimates

```{glue:} plot_mack
```

+++
**Algorithms**: [Chainladder](methods:chainladder), [CapeCod](methods:capecod), …


---
:header: bg-jb-two

**[Adjustments](adjustments)**
^^^
Common actuarial data adjustments



**Applications**: Simulation, trending, on-leveling

```{glue:} plot_stochastic_bornferg
```

+++
**Classes**: [BootstrapODPSample](adjustments:bootstrapodpsample), [Trend](adjustments:trend), …

---
:header: bg-jb-three

**[Workflow](workflow)**
^^^

Workflow tools for complex analyses

**Application**: Scenario testing, ensembling

```{glue:} plot_voting_chainladder
```

+++
**Utilities**: [Pipeline](workflow:pipeline), [VotingChainladder](workflow:votingchainladder), …
:::




