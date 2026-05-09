================================================================
Chapter 6 - The Development Triangle as a Diagnostic Tool
================================================================

This chapter dives deeper into understanding the triangle. We will demonstrate how to manipulate a ``Triangle`` in the chainladder package by recreating the various tables in this chapter. 

.. doctest::

    >>> import numpy as np
    >>> import pandas as pd
    >>> import chainladder as cl
    >>> tri = cl.load_sample('friedland_xyz_auto_bi')

Table 1 - Summary of Earned Premium and Rate Changes
#######################################################

WIP

Table 2 - Reported Claim Development Triangle
##################################################

We don't need the full triangle for this chapter. Here is how to filter a triangle based on accident year and development age. 

.. doctest::

    >>> tri = tri[tri.origin >= '2002'][tri.development <= 84]
    >>> tri['Reported Claims']

Table 3 - Paid Claim Development Triangle
##################################################

.. doctest::

    >>> tri['Paid Claims']
