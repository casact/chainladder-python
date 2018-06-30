# -*- coding: utf-8 -*-
"""
The Triangle Module includes the Triangle class.  This base structure of the
class is a pandas.DataFrame.
"""

import pandas as pd
import numpy as np
from chainladder.UtilityFunctions import Plot
from bokeh.palettes import Spectral10
import copy

class Triangle():
    """Triangle class is the basic data representation of an actuarial triangle.

    Historical insurance data is often presented in form of a triangle structure, showing
    the development of claims over time for each exposure (origin) period. An origin
    period could be the year the policy was written or earned, or the loss occurrence
    period. Of course the origin period doesn’t have to be yearly, e.g. quarterly or
    6
    monthly origin periods are also often used. The development period of an origin
    period is also called age or lag. Data on the diagonals present payments in the
    same calendar period. Note, data of individual policies is usually aggregated to
    homogeneous lines of business, division levels or perils.
    Most reserving methods of the ChainLadderpackage expect triangles as input data
    sets with development periods along the columns and the origin period in rows. The
    package comes with several example triangles.
    `Proper Citation Needed... <https://github.com/mages/ChainLadder>`_

    Parameters:
        data : pandas.DataFrame
            A DataFrame representing the triangle
        origin : str or list
            An array representing the origin period of the triangle.
        development : str or list
            An array representing the development period of the triangle. In triangle form,
            the development periods must be the columns of the dataset
        values : str
            A string representing the column name of the triangle measure if the data is in
            tabular form.  Otherwise it is ignored.

    Attributes:
        data : pandas.DataFrame
            A DataFrame representing the triangle
        origin_grain : str
            The level of granularity of the development of the triangle.
        development_grain : str
            The level of granularity of the development of the triangle.


    """

    def __init__(self, data=None, origin=None, development=None, values=None):
        if type(data) is not pd.DataFrame:
            raise TypeError(str(type(data)) + \
                            ' is not a valid datatype for the Triangle class.  Currently only DataFrames are supported.')
        self.data = data.copy()
        if data.shape[1]>= data.shape[0]:
            self.data.index.name = 'orig'
            self.data.reset_index(inplace=True)
            self.__set_period(axis=0, fields=['orig'])
            self.origin_grain = self.__get_period_grain(axis=0)
            self.data.set_index('origin', inplace=True)
            self.development_grain = self.__get_period_grain()
        else:
            development = list(development) if type(development) is not str else [development]
            origin = list(origin) if type(origin) is not str else [origin]
            data['values'] = data[values]
            self.data = data[origin + development + ['values']].copy()
            self.__set_period(axis=0, fields=origin)
            self.__set_period(axis=1, fields=development)
            self.origin_grain = self.__get_period_grain(axis=0)
            self.development_grain = self.__get_period_grain(axis=1)
            self.data = self.__data_as_triangle()

    def __set_period(self, axis, fields):
        '''For tabular form, this will take a set of origin or development
        column(s) and convert them to dates.
        '''
        target_field = 'origin' if axis == 0 else 'development'
        if target_field in fields:
            raise ValueError('The column names `origin` and/or `development` are reserved.')
        self.data[target_field] = ''
        for item in fields:
            self.data[target_field] = self.data[target_field] + self.data[item].astype(str)
        # Use pandas to infer date
        datetime_arg = self.data[target_field].astype(str).str.strip()
        date_inference_list = [{'arg':datetime_arg, 'format':'%Y%m'},
                               {'arg':datetime_arg, 'infer_datetime_format':True},]
        for item in date_inference_list:
            try:
                self.data[target_field] = pd.to_datetime(**item)
                break
            except:
                pass
        self.data.drop(fields, axis=1,inplace=True)

    def __get_period_grain(self, axis = None):
        '''For tabular form. Given date granlarity, it will return Y = Year,
        Q = Quarter or M = Month grain
        '''
        if axis is not None:
            target_field = 'origin' if axis == 0 else 'development'
            months = self.data[target_field].dt.month.unique()
            quarters = self.data[target_field].dt.quarter.unique()
            grain = 'Y' if len(quarters) == 1 else 'Q'
            grain = grain if len(months) == len(quarters) else 'M'
        else:
            vertical_diff = max(len(self.data.iloc[:,-2].dropna()) - len(self.data.iloc[:,-1].dropna()),1)
            horizontal_diff = max(len(self.data.iloc[-2].dropna()) - len(self.data.iloc[-1].dropna()),1)
            dgrain_dict = {(1,4):'Q', (1,12):'M',(4,1):'Y',(1,3):'M',(12,1):'Y',(3,1):'Q'}
            grain = self.origin_grain if vertical_diff == horizontal_diff \
                    else dgrain_dict[(vertical_diff,horizontal_diff)]
        return grain

    def __get_dev_lag(self):
        ''' For tabular format, this will convert the origin/development difference
        to a development lag '''
        year_diff = self.data['development'].dt.year - self.data['origin'].dt.year
        quarter_diff = self.data['development'].dt.quarter - self.data['origin'].dt.quarter
        month_diff = self.data['development'].dt.month - self.data['origin'].dt.month
        dev_lag_dict = {'Y':year_diff + 1,
                        'Q':year_diff * 4 + quarter_diff + 1,
                        'M':year_diff * 12 + month_diff + 1}
        dev_lag = dev_lag_dict[self.development_grain].rename('dev_lag')
        return pd.concat([self.data[['origin','development']], dev_lag], axis=1)

    def __get_development(self, obj = None):
        ''' For triangle format, this will convert the origin/dev_lag difference
        to a development date '''
        if obj is None:
            obj = self.data
        if self.development_grain == 'Q':
            development = obj.reset_index()['origin'].values.astype('datetime64[M]') + \
                    ((obj.reset_index()['dev_lag']*3)-3).values.astype('timedelta64[M]')
        else:
            development = obj.reset_index()['origin'].values.astype('datetime64[' + self.development_grain + ']') + \
                    (obj.reset_index()['dev_lag']-1).values.astype('timedelta64[' + self.development_grain + ']')
        return pd.concat([obj.reset_index()[['origin','dev_lag']], pd.Series(development, name='development')], axis=1)

    def __repr__(self):
        return str(self.__format())

    def to_clipboard(self):
        '''pass-through of pandas DataFrame.to_clipboard() functionality '''
        return self.__format().to_clipboard()

    def __format(self):
        '''The purpose of this method is to display the dates at the grain
        of the triangle, i.e. don't dispay months if quarter is lowest grain'''
        temp = self.data.copy().reset_index()
        temp['origin'] = pd.PeriodIndex(temp['origin'], freq=self.origin_grain)
        temp.set_index('origin', inplace=True)
        return temp


    def __data_as_triangle(self):
        """Method to convert tabular form to triangle form.

        Returns:
            Updated instance `data` parameter if inplace is set to True otherwise it returns a pandas.DataFrame

        unit tests for success:
            1. must retain the original origin date
        """
        dev_lag = self.data.merge(self.__get_dev_lag(), how='inner', on=['origin','development'])
        origin_dict = {'Y':self.data['origin'].dt.strftime('%Y'),
                       'Q':self.data['origin'].dt.year.astype(str) +'-'+ 'Q' + self.data['origin'].dt.quarter.astype(str),
                       'M':self.data['origin'].dt.strftime('%Y-%b')}
        tri = pd.pivot_table(dev_lag, values='values', index='origin',
                             columns='dev_lag', aggfunc="sum").sort_index()
        tri.columns = [item for item in tri.columns]
        #fills in lower diagonal with NANs
        tri = self.__fill_inner_diagonal_nans(tri)
        return tri

    def data_as_table(self):
        """Method to convert triangle form to tabular form.

        Arguments:
            inplace: bool
                Set to True will update the instance data attribute inplace

        Returns:
            Updated instance `data` parameter if inplace is set to True otherwise it returns a pandas.DataFrame
        """
        idx = list(self.data.index.names)
        data = pd.melt(self.data.reset_index(),id_vars=idx, value_vars=self.data.columns,
                       var_name='dev_lag', value_name='values').dropna().set_index(idx).sort_index().reset_index()
        data = data.merge(self.__get_development(data), how='inner', on=['origin','dev_lag']).drop('dev_lag',axis=1).sort_values(['origin','development'])
        return data

    def __fill_inner_diagonal_nans(self, data):
        '''purpose of this method is to provide data clean-up with missing Values
        in inner diagonals of our data. This occurs in practice where the development
        periods avaialble do not go as far back as the origin periods.
        '''
        cols = data.columns
        index = data.index
        temp = np.array(data)
        key = []
        temp_array = np.nan_to_num(np.array(data))
        for i in range(len(cols)):
            x = np.where(np.isfinite(temp[:,i][::-1]))[0][0]
            key.append(x)
            if x > 0: temp_array[:,i][-x:] = np.nan
        return pd.DataFrame(temp_array, index=index, columns=cols)

    def cum_to_incr(self, inplace=False):
        """Method to convert an cumulative triangle into a incremental triangle.  Note,
        the triangle must be in triangle form.

        Arguments:
            inplace: bool
                Set to True will update the instance data attribute inplace

        Returns:
            Updated instance `data` parameter if inplace is set to True otherwise it returns a pandas.DataFrame

        """

        if inplace == True:
            data = np.array(self.data)
            incr = np.concatenate((data[:,0].reshape(data.shape[0],1),
                                   data[:,1:]-data[:,:-1]), axis=1)
            self.data = self.data*0 + incr
            return self
        if inplace == False:
            new_instance = copy.deepcopy(self)
            return new_instance.cum_to_incr(inplace=True)

    def incr_to_cum(self, inplace=False):
        """Method to convert an incremental triangle into a cumulative triangle.  Note,
        the triangle must be in triangle form.

        Arguments:
            inplace: bool
                Set to True will update the instance data attribute inplace

        Returns:
            Updated instance `data` parameter if inplace is set to True otherwise it returns a pandas.DataFrame

        """

        if inplace == True:
            self.data = self.data.cumsum(axis=1)
            return self
        if inplace == False:
            new_instance = copy.deepcopy(self)
            return new_instance.incr_to_cum(inplace=True)

    def get_latest_diagonal(self):
        """Method to return the latest diagonal of the triangle.

        Returns: pandas.Series

        """

        data = self.data_as_table()
        return data[data['development'] == data['development'].max()] \
                    .sort_values('origin') \
                    .set_index('origin') \
                    .drop('development',axis=1)

    def __add__(self, other):
        return Triangle(self.data + other.data)

    def __radd__(self, other):
        return self if other == 0 else self.__add__(other)

    def __sub__(self, other):
        return Triangle(self.data - other.data)

    def __mul__(self, other):
        return Triangle(self.data * other.data)

    def __truediv__(self, other):
        return Triangle(self.data / other.data)

    def grain(self, grain='', incremental=False, inplace=False):
        if inplace == True:
            if grain == '': grain = 'O' + self.origin_grain + 'D' + self.development_grain
            # At this point we need to convert to tabular format
            ograin = grain[1:2]
            if incremental == True: self.incr_to_cum(inplace=True)
            temp = self.data.copy()
            if self.origin_grain != ograin:
                self.origin_grain = ograin
                temp = self.data_as_table()
                temp['origin'] = pd.PeriodIndex(temp['origin'],freq=ograin).to_timestamp()
                temp = pd.pivot_table(temp,index=['origin','development'],values='values', aggfunc='sum').reset_index()
                col_dict = {'origin':'o','development':'d','values':'v'}
                temp.columns = [col_dict.get(item, item) for item in temp.columns]
                temp = Triangle(temp, origin='o',development='d',values='v')
            # At this point we need to be in triangle format
            self.data = temp
            dgrain = grain[-1]
            if self.development_grain != dgrain:
                init_lag = self.data.iloc[-1].dropna().index[-1]
                final_lag = self.data.iloc[0].dropna().index[-1] + 1
                freq = {'M':{'Y':12,'Q':3,'M':1},
                        'Q':{'Y':4,'Q':1}}
                self.data = self.data[[item for item in range(init_lag,final_lag,freq[self.development_grain][dgrain])]]

                self.development_grain = dgrain
            if incremental == True:
                self.cum_to_incr(inplace=True)
            return self
        if inplace == False:
            new_instance = copy.deepcopy(self)
            return new_instance.grain(grain=grain, incremental=incremental, inplace=True)

    def plot(self, ctype='m', plots=['triangle'], **kwargs):
        """ Method, callable by end-user that renders the matplotlib plots.

        Arguments:
            plots: list[str]
                A list of strings representing the charts the end user would like
                to see.  If ommitted, all plots are displayed.  Available plots include:
                    ============== =================================================
                    Str            Description
                    ============== =================================================
                    triangle       Line chart of origin period x development period
                    ============== =================================================

        Returns:
            Renders the matplotlib plots.

        """
        my_dict = []
        plot_dict = self.__get_plot_dict()
        for item in plots:
            my_dict.append(plot_dict[item])
        return Plot(ctype, my_dict, **kwargs).grid

    def __get_plot_dict(self):
        '''Should probably put the configuration in a yaml or json file
        '''
        xvals = [i+1 for i in range(len(self.data.columns))]
        plot_dict = \
            {'triangle': \
                {'Title':'Latest Triangle Data',
                 'XLabel':'Development Period',
                 'YLabel':'Values',
                 'chart_type_dict': \
                    {'type':['line'],
                     'mtype':['line'],
                     'rows':[len(self.data)],
                     'x':[xvals],
                     'y':[self.data],
                     'line_dash':['solid'],
                     'line_width':[4],
                     'alpha':[1],
                     'line_cap':['round'],
                     'line_join':['round'],
                     'color':[Spectral10*int(len(self.data)/10+1)],
                     'label':[[str(item) for item in self.data.index]],
                     'linestyle':['-'],
                     'linewidth':[5],
                     }
                }
            }
        return plot_dict
