from linkcode import linkcode_resolve

def test_linkcode_resolve_url_lines() -> None:
    """
    Tests the linkcode_resolve function. Ideally we want:

     - A URL that points to GitHub
     - Starting and ending line numbers in the URL.
    """
    url: str = linkcode_resolve(
        domain='py',
        info={
            'module': 'chainladder',
            'fullname': 'Benktander'
        }
    )

    assert url is not None
    assert url.startswith('https://github.com/casact/chainladder-python/blob/master/')

    # Extract the line numbers.
    lines: str = url.split('#')[1]
    start, end = [int(x[1:]) for x in lines.split('-')]

    # Check the line numbers.
    assert start > 0
    assert end > start
