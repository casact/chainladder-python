# chainladder documentation

This folder contains everything that builds the [chainladder-python docs site](https://chainladder-python.readthedocs.io/). It is organised as a [Jupyter Book](https://jupyterbook.org/) v1.x project on top of Sphinx, with the API reference produced by `sphinx.ext.autosummary` from the package docstrings.

If you only want to read the docs, use the published site. This README is for contributors editing the docs.

## Building the docs locally

From the repository root:

```bash
pip install -e .[docs]
cd docs
jb build .
```

The rendered site is written to `docs/_build/html/`. Open `docs/_build/html/index.html` in a browser to review changes.

The underlying Sphinx targets are also available through `make html`, `make doctest`, `make linkcheck` (or `make.bat` on Windows) if you want finer-grained control.

## Source files (edit these)

| File / folder | Purpose |
| --- | --- |
| `intro.md` | Landing page of the docs site. |
| `_toc.yml` | Book structure and page ordering. |
| `_config.yml` | Jupyter Book configuration (title, theme, Sphinx extensions). |
| `conf.py` | Sphinx configuration (autosummary, intersphinx, numpydoc, etc.). |
| `getting_started/` | Install, onboarding, and Quickstart notebooks. |
| `user_guide/` | Long-form topic notebooks (triangle, development, tails, methods, adjustments, workflow, utils). |
| `library/api.md` | API reference index. Lists every public class/function via `autosummary` directives. |
| `library/releases.md` | Release notes. |
| `gallery/` | Example notebooks rendered into the gallery. |
| `images/` | Static images referenced from the notebooks and markdown. |
| `Makefile`, `make.bat` | Sphinx build entry points. |

To document a new public class or function, add it to the appropriate `autosummary` block in `library/api.md` and write the docstring in the source module under `chainladder/`. The API page itself is generated from the docstring; you do not author per-class `.rst` files by hand.

## Generated files (do not edit)

These are produced by the build and should not be edited directly. Edits will be overwritten.

| Path | Produced by |
| --- | --- |
| `_build/` | `jb build .` / `sphinx-build`. Gitignored. |
| `library/generated/*.rst` | `sphinx.ext.autosummary` (`autosummary_generate = True` in `conf.py`, driven by the `:toctree: generated/` directives in `library/api.md`). |
