# Contributing

**`chainladder-python`** welcomes volunteers at all levels, whether you are new to actuarial reserving, Python, or both. Feedback, questions, suggestions, and contributions are all highly encouraged.

---

## Why Contribute

- Gain practical experience with Python, software development, and actuarial methods.  
- Meet like-minded actuaries that prefer open-source software.
- Help advance open-source actuarial research and reproducible methodologies.  
- Contributing is rewarding and fun!  

---

## How to Contribute

You can help improve and shape the project in many ways:  

1. Submit bugs or enhancement requests via the [issue tracker](https://github.com/casact/chainladder-python/issues).  
2. Volunteer to implement code changes for existing issues.  
3. Ask questions or discuss ideas on the [discussion forum](https://github.com/casact/chainladder-python/discussions).  
4. Improve documentation where it is unclear.  
5. Create new examples or tutorials in the [examples section](https://chainladder-python.readthedocs.io/en/latest/gallery/index.html).  

---

## Contributors Working Group

We also have a Contributors Working Group that meets for one hour approximately every two weeks. We are part of the CAS volunteer groups and are supported by the CAS. If you are interested, you are welcome to join us through the Open-Source Projects Working Group.

We currently meet on Fridays 11:00am - Noon ET, but we periodically adjust meeting times to accommodate contributors across time zones around the globe. Our meetings are relaxed and collaborative, with discussions around milestones, package design ideas, open issues, and other behind-the-scenes work on GitHub.

We welcome contributors of all skill levels, including CAS members, its affiliate members, industry researchers, educators, students, and CAS candidates. To join us, you may respond to the CAS annual Volunteer Interest and Participation (VIP) Survey, reach out to a CAS staff persons (Heather Davis <hdavis@casact.org>, Elizabeth Smith <esmith@casact.org>), or the working group chair (Kenneth Hsu <kennethshsu@gmail.com>), and we will be happy to welcome you into the group.

---

## API Guidelines

`chainladder-python` is designed to follow the style of **pandas** and **scikit-learn**. This ensures a consistent API and simplifies new development:  

1. Estimators and transformers should follow the [scikit-learn estimator API](https://scikit-learn.org/stable/developers/develop.html).  
2. `Triangle` methods should align with **pandas** (then **NumPy**) naming and signature conventions. Domain-specific methods with no equivalent should follow [PEP8](https://www.python.org/dev/peps/pep-0008/#method-names-and-instance-variables).  
3. `Triangle` methods are non-mutating by default. Mutation is allowed only via an `inplace` argument.  

For domain-specific exceptions, discuss on the [issue tracker](https://github.com/casact/chainladder-python/issues).

---

## Development Environment

After forking the repository, you can set up a development environment using `uv`:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to your PATH
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Navigate to your working directory
cd chainladder-python

# Create virtual environment and install all dependencies
uv sync --extra all

# Activate the environment
uv run python # or uv run jupyter-lab
```

This will install the package in editable mode with all development dependencies. After finishing work, deactivate:

```bash
deactivate
```

---

## Documentation

Documentation is built with **Jupyter Book**. From your development environment:  

```bash
cd docs
jb build .
```

The generated site is in `_build/html`. Documentation is maintained in:  

1. Docstrings within the code (hosted in API reference).  
2. User Guide notebooks (`/docs/modules`).  
3. Tutorial notebooks (`/docs/tutorials`).  

Contributions to documentation are especially helpful for new users.

---

## Pull Requests (PRs)

**Guidelines for PRs:**  

**PRs must:**  
- Pass all existing unit tests  
- Undergo independent peer review  

**PRs are encouraged to:**  
- Be small, focused, and modular  
- Link to relevant Issue ticket(s)  
- Include docstring updates for any code changes  
- Update the documentation site for corresponding changes  
- Follow established naming conventions  
- Include new unit tests with reasonable coverage  

All PRs should be run locally before submission:  

```bash
pytest chainladder
```

Large or unfocused PRs may delay merging. Each PR should address a single issue or feature to maintain clarity and quality.
