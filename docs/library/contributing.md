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

We also have a **Contributors Working Group** that meets approximately every two weeks for one hour. The current meeting time is **Fridays from 9:00–10:00 AM PT / 12:00–1:00 PM ET**.  

During these meetings, we discuss milestones, package design philosophies, outstanding issues, and other behind-the-scenes topics related to our work on GitHub.  

If you're interested in joining, please contact one of the core developers. We welcome contributors of all skill levels and encourage your involvement.

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
uv run python
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
