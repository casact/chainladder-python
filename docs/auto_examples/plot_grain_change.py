"""
============================
Changing grain of a triangle
============================

If your triangle has origin and development grains that are more frequent then
yearly, you can easily swap to a higher grain using the `grain` method of the
:class:`Triangle`.
In this example, we will convert an Origin Year/Development Quarter (OYDQ)
triangle into an Origin Year/Development Year triangle.  The `grain` method
recognizes Yearly (Y), Quarterly (Q), and Monthly (M) grains for both the
origin period and development period.
"""


import chainladder as cl

# The base Triangle Class:
cl.Triangle


quarterly = cl.load_dataset('quarterly')
print(quarterly)
print()
print(quarterly.grain('OYDY'))
