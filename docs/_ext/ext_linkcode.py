from __future__ import annotations

import inspect
import importlib
import os

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

def linkcode_resolve(
        domain: str,
        info: dict
) -> str | None:
    """
    Implementation of linkcode_resolve from the Sphinx extension sphinx.ext.linkcode. For more information, see:

    https://www.sphinx-doc.org/en/master/usage/extensions/linkcode.html

    Extracts objects and their line numbers from the chainladder package. These are then used to construct
    a URL that links to the object's source code on GitHub.

    To debug, try supplying the following arguments:
    domain = 'py',
    info = {'module': 'chainladder', 'fullname': 'Benktander'}

    Parameters
    ----------
    domain: str
        Specifies the language the object is in.
    info: dict
        A dictionary containing information about the object.

    Returns
    -------
        A URL to the object's source code on GitHub.
    """

    # Base path pointing to GitHub source code.
    url_base: str = 'https://github.com/casact/chainladder-python/blob/master/'

    # Get the language information.
    if domain != 'py':
        return None
    if not info['module']:
        return None

    # Extract the module.
    obj: ModuleType = importlib.import_module(info['module'])

    # Extract the part.
    for part in info['fullname'].split('.'):
        obj: type = getattr(obj, part)

    # Unwrap any decorators.
    obj: type = inspect.unwrap(obj)

    # Get the path of the source file.
    source_file: str = inspect.getsourcefile(obj)

    # If the source file cannot be found, return the URL pointing to a .py file named after the module.
    if source_file is None:
        filename: str = info['module'].replace('.', '/')
        return url_base + "%s.py" % filename

    # Extract the lines and locate the starting line position.
    source_lines, start_line = inspect.getsourcelines(obj)

    # Get the root path.
    repo_root: str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..')
    )

    # Construct a relative path using the source file path.
    rel_path: str = os.path.relpath(source_file, repo_root)

    # Extract the ending line.
    end_line: int = start_line + len(source_lines) - 1

    # Construct the URL by appending the relative path, starting, and ending lines to the base URL.
    return url_base + '%s#L%d-L%d' % (
        rel_path, start_line, end_line
    )

def setup(app) -> None:
    """
    Function required to activate the local extension. Reference:

    https://www.sphinx-doc.org/en/master/development/tutorials/extending_syntax.html#using-the-extension
    """
    app.config.linkcode_resolve = linkcode_resolve
