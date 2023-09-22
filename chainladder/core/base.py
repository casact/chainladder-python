# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


def is_chainladder(estimator):
    """Return True if the given estimator is a chainladder based method.
    Parameters
    ----------
    estimator : object
        Estimator object to test.
    Returns
    -------
    out : bool
        True if estimator is a chainladder based method and False otherwise.
    """
    return getattr(estimator, "_estimator_type", None) == "chainladder"
