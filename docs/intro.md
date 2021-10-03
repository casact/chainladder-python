# Welcome to Chainladder

```{DANGER}
These docs are under major development. If you want to visit our stable docs, please go [here](https://chainladder-python.readthedocs.io/en/stable/)
```


## Casualty Loss Reserving in Python

* Simple and efficient tools for actuarial loss reserving
* Designed with practical workflows in mind
* Looks like Pandas and Scikit-Learn, the tools you love!
* Open source, commercially usable - MPL-2.0 license

Here are a few links to help you get started.

:::{panels}
:container: +full-width
:column: col-lg-4 px-2 py-2
---
:header: bg-jb-one

**[Triangles](modules/triangle.rst)**
^^^
Data object to manage and manipulate reserving data

**Application**: Extend pandas syntax to manipulate reserving triangles

**Classes**: **[Triangle](modules/triangle)**,...

---
:header: bg-jb-two

**[Development](modules/development.rst)**
^^^
Tooling to generate loss development

**Applications**: comprehensive library of development techniques

**Algorithms**: [Development](development:development), [MunichAdjustment](modules/development/development), [ClarkLDF](development), …
---
:header: bg-jb-three

**[Tail Estimation](modules/tails.rst)**
^^^
Extrapolate development patterns beyond the known data.

**Applications**: Long-tailed lines of business

**Algorithms**: TailCurve, TailConstant, TailBondy, …

:::


:::{panels}
:container: +full-width
:column: col-lg-4 px-2 py-2
---
:header: bg-jb-one
**[IBNR Models](modules/methods.rst)**
^^^

Generate IBNR estimates and associated statistics

**Applications**: constructing reserve estimates

**Algorithms**: Chainladder, BornhuetterFerguson, CapeCod, …


---
:header: bg-jb-two

**[Adjustments](modules/adjustments.rst)**
^^^
Common actuarial data adjustments

**Applications**: Simulation, trending, on-leveling

**Classes**: BootstrapODPSample, BerquistSherman, Trend,…

---
:header: bg-jb-three

**[Workflow](modules/workflow.rst)**
^^^

Workflow tools for complex analyses

**Application**: scenario testing, simulation, ensembling

**Utilities**: Pipeline, VotingChainladder, …
:::
