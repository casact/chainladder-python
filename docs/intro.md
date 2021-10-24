# Welcome to Chainladder


Chainladder-python is a python library for casualty loss reserving. By loss
reserving, we are refering to the work that would be conducted by an actuary. It is:

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

**Algorithms**: [Development](development:development), [MunichAdjustment](development:munichadjustment), [ClarkLDF](development:clarkldf), …
---
:header: bg-jb-three

**[Tail Estimation](modules/tails.rst)**
^^^
Extrapolate development patterns beyond the known data.

**Applications**: Long-tailed lines of business

**Algorithms**: [TailCurve](tails:tailcurve), [TailConstant](tails:tailconstant), [TailBondy](tails:tailbondy), …

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

**Algorithms**: [Chainladder](methods:chainladder), [BornhuetterFerguson](methods:bornhuetterferguson), [CapeCod](methods:capecod), …


---
:header: bg-jb-two

**[Adjustments](modules/adjustments.rst)**
^^^
Common actuarial data adjustments

**Applications**: Simulation, trending, on-leveling

**Classes**: [BootstrapODPSample](adjustments:bootstrapodpsample), [BerquistSherman](adjustments:berquistsherman), [Trend](adjustments:trend),…

---
:header: bg-jb-three

**[Workflow](modules/workflow.rst)**
^^^

Workflow tools for complex analyses

**Application**: scenario testing, simulation, ensembling

**Utilities**: [Pipeline](workflow:pipeline), [VotingChainladder](workflow:votingchainladder), …
:::
