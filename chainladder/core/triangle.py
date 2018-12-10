'''
Triangle class creates multiple actuarial triangles in an easy to manage class.
The API borrows heavily from pandas for the purpose of indexing, slicing, and
accessing individual triangles within the broader class.

The core data structure at the heart of the Triangle class is a 4D numpy array
with dimensions defined as:
    Dimension 0: represents key dimensions or the lowest grain(s) at which you
                 want to manage the triangle, e.g State, Company, etc. The
                 grain supports multiple key dimensions that will behave like a
                 pandas.multiIndex

    Dimension 1: represents value dimensions or numeric data to be represented
                 in each triangle, e.g. Paid, Incurred, etc.

    Dimension 2: represents the origin dimension which will be stored as a date
                 e.g. Accident Month, Report Year, Policy Quarter, etc.

    Dimension 3: represents the development dimension which will be store
                 e.g. Valuation Month, Valuation Year, Valuation Quarter, etc.

Dimensions 0 and 1 are accessed like a pandas Dataframe.  You can think about
the 4d structure as a 2D Dataframe where each element (row, col) is its own 2D
triangle. The most common operations in pandas are included and allows for
powerful manipulation. For example:

    my_triangle['incurred'] = my_triangle['paid'] + my_triangle['reserve']

This snippet would extend the values dimension of the my_triangle instance for
all key, origin and development dimensions.

'''
import pandas as pd
import numpy as np
import copy
from chainladder.core.base import TriangleBase, check_triangle_postcondition


class Triangle(TriangleBase):
    def __init__(self, data=None, origin=None, development=None,
                 values=None, keys=None):
        # Sanitize Inputs
        values = [values] if type(values) is str else values
        origin = [origin] if type(origin) is str else origin
        if type(development) is str:
            development = [development]
        if keys is None:
            keys = ['Total']
            data_agg = data.groupby(origin+development).sum().reset_index()
            data_agg[keys[0]] = 'Total'
        else:
            data_agg = data.groupby(origin+development+keys) \
                           .sum().reset_index()

        # Convert origin/development to dates
        origin_date = TriangleBase.to_datetime(data_agg, origin)
        self.origin_grain = TriangleBase.get_grain(origin_date)
        # These only work with valuation periods and not lags
        development_date = TriangleBase.to_datetime(data_agg, development)
        self.development_grain = TriangleBase.get_grain(development_date)

        # Prep the data for 4D Triangle
        data_agg = self.get_axes(data_agg, keys, values,
                                 origin_date, development_date)
        data_agg = pd.pivot_table(data_agg, index=keys+['origin'],
                                  columns='development', values=values,
                                  aggfunc='sum')
        # Assign object properties
        self.kdims = np.array(data_agg.index.droplevel(-1).unique())
        self.odims = np.array(data_agg.index.levels[-1].unique())
        self.ddims = np.array(data_agg.columns.levels[-1].unique())
        self.vdims = np.array(data_agg.columns.levels[0].unique())
        self.valuation_date = development_date.max()
        self.key_labels = keys
        self.iloc = TriangleBase.Ilocation(self)
        self.loc = TriangleBase.Location(self)
        # Create 4D Triangle
        triangle = \
            np.array(data_agg).reshape(len(self.kdims), len(self.odims),
                                       len(self.vdims), len(self.ddims))
        triangle = np.swapaxes(triangle, 1, 2)
        # Set all 0s to NAN for nansafe ufunc arithmetic
        triangle[triangle == 0] = np.nan
        self.triangle = triangle
        # Used to show NANs in lower part of triangle
        self.nan_override = False

    @check_triangle_postcondition
    def grain(self, grain='', incremental=False, inplace=False):
        ''' TODO - Make incremental work '''
        if inplace:
            origin_grain = grain[1:2]
            development_grain = grain[-1]
            new_tri, o = self._set_ograin(grain=grain, incremental=incremental)
            # Set development Grain
            dev_grain_dict = {'M': {'Y': 12, 'Q': 3, 'M': 1},
                              'Q': {'Y': 4, 'Q': 1}}
            keeps = dev_grain_dict[self.development_grain][development_grain]
            keeps = self.ddims % keeps == 0
            new_tri = new_tri[:, :, :, keeps]
            self.ddims = self.ddims[keeps]
            self.odims = np.unique(o)
            self.origin_grain = origin_grain
            self.development_grain = development_grain
            self.triangle = new_tri
            self.triangle = self._slide(self.triangle, direction='l')
            self.triangle[self.triangle == 0] = np.nan
            return self
        else:
            new_obj = copy.deepcopy(self)
            new_obj.grain(grain=grain, incremental=incremental, inplace=True)
            return new_obj


class Exposure(TriangleBase):
    def __init__(self, data=None, origin=None, values=None, keys=None):
        # Sanitize Inputs
        values = [values] if type(values) is str else values
        origin = [origin] if type(origin) is str else origin
        if keys is None:
            keys = ['Total']
            data_agg = data.groupby(origin).sum().reset_index()
            data_agg[keys[0]] = 'Total'
        else:
            data_agg = data.groupby(origin+keys) \
                           .sum().reset_index()

        # Convert origin/development to dates
        origin_date = TriangleBase.to_datetime(data_agg, origin)
        self.origin_grain = TriangleBase.get_grain(origin_date)
        development_date = origin_date
        self.development_grain = self.origin_grain

        # Prep the data for 4D Triangle
        data_agg = self.get_axes(data_agg, keys, values,
                                 origin_date, development_date)
        data_agg = pd.pivot_table(data_agg, index=keys+['origin'],
                                  values=values, aggfunc='sum')
        # Assign object properties
        self.kdims = np.array(data_agg.index.droplevel(-1).unique())
        self.odims = np.array(data_agg.index.levels[-1].unique())
        self.ddims = np.array(['Exposure'])
        self.vdims = np.array(data_agg.columns.unique())
        self.valuation_date = development_date.max()
        self.key_labels = keys
        self.iloc = TriangleBase.Ilocation(self)
        self.loc = TriangleBase.Location(self)
        # Create 4D Triangle
        triangle = \
            np.array(data_agg).reshape(len(self.kdims), len(self.odims),
                                       len(self.vdims), len(self.ddims))
        triangle = np.swapaxes(triangle, 1, 2)
        # Set all 0s to NAN for nansafe ufunc arithmetic
        triangle[triangle == 0] = np.nan
        self.triangle = triangle

    def grain(self, grain='', incremental=False, inplace=False):
        ''' TODO - Make incremental work '''
        if inplace:
            origin_grain = grain[1:2]
            development_grain = grain[-1]
            new_tri, o = self._set_ograin(grain=grain, incremental=incremental)
            self.ddims = ['Exposure']
            self.odims = np.unique(o)
            self.origin_grain = origin_grain
            self.development_grain = development_grain
            self.triangle = new_tri
            self.triangle = self._slide(self.triangle, direction='l')
            return self
        else:
            new_obj = copy.deepcopy(self)
            new_obj.grain(grain=grain, incremental=incremental, inplace=True)
            return new_obj

    def latest_diagonal(self):
        return self
