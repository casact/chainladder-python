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
        dataform : str
            A string value that takes on one of two values ['triangle' and 'tabular']

    Attributes:
        data : pandas.DataFrame
            A DataFrame representing the triangle
        origin_grain : str
            The level of granularity of the development of the triangle.
        development_grain : str
            The level of granularity of the development of the triangle.
        dataform : str
            Refer to parameter value.

    """

    def __init__(self, data=None, origin=None, development=None, values=None, dataform=None):
        if type(data) is not pd.DataFrame:
            raise TypeError(str(type(data)) + \
                            ' is not a valid datatype for the Triangle class.  Currently only DataFrames are supported.')
        self.data = copy.deepcopy(data)
        if dataform is None:
            if data.shape[1]>= data.shape[0]:
                self.dataform = 'triangle'
            else:
                self.dataform = 'tabular'
        if self.dataform == 'tabular':
            development = list(development) if type(development) is not str else [development]
            origin = list(origin) if type(origin) is not str else [origin]
            data['values'] = data[values]
            self.data = data[origin + development + ['values']].copy()
            self.__set_period(axis=0, fields=origin)
            self.__set_period(axis=1, fields=development)
            self.origin_grain = self.__get_period_grain(axis=0)
            self.development_grain = self.__get_period_grain(axis=1)
        else:
            self.data.index.name = 'orig'
            self.data.reset_index(inplace=True)
            self.__set_period(axis=0, fields=['orig'])
            self.origin_grain = self.__get_period_grain(axis=0)
            self.data.set_index('origin', inplace=True)
            self.development_grain = self.__get_period_grain()

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

    def __get_development(self):
        ''' For triangle format, this will convert the origin/dev_lag difference
        to a development date '''
        if self.development_grain == 'Q':
            development = self.data.reset_index()['origin'].values.astype('datetime64[M]') + \
                    ((self.data.reset_index()['dev_lag']*3)-3).values.astype('timedelta64[M]')
        else:
            development = self.data.reset_index()['origin'].values.astype('datetime64[' + self.development_grain + ']') + \
                    (self.data.reset_index()['dev_lag']-1).values.astype('timedelta64[' + self.development_grain + ']')
        return pd.concat([self.data.reset_index()[['origin','dev_lag']], pd.Series(development, name='development')], axis=1)

    def __repr__(self):
        return str(self.__format())

    def to_clipboard(self):
        '''pass-through of pandas DataFrame.to_clipboard() functionality '''
        return self.__format().to_clipboard()

    def __format(self):
        '''The purpose of this method is to display the dates at the grain
        of the triangle, i.e. don't dispay months if quarter is lowest grain'''
        temp = self.data.copy()
        if self.dataform == 'triangle':
            temp = temp.reset_index()
            temp['origin'] = pd.PeriodIndex(temp['origin'], freq=self.origin_grain)
            temp.set_index('origin', inplace=True)
        if self.dataform == 'tabular':
            temp['origin'] = pd.PeriodIndex(temp['origin'], freq=self.origin_grain)
            temp['development'] = pd.PeriodIndex(temp['development'], freq=self.development_grain)
        return temp


    def data_as_triangle(self, inplace=False):
        """Method to convert tabular form to triangle form.

        Arguments:
            inplace: bool
                Set to True will update the instance data attribute inplace

        Returns:
            Updated instance `data` parameter if inplace is set to True otherwise it returns a pandas.DataFrame

        unit tests for success:
            1. must retain the original origin date
        """
        if self.dataform == 'tabular':
            dev_lag = self.data.merge(self.__get_dev_lag(), how='inner', on=['origin','development'])
            origin_dict = {'Y':self.data['origin'].dt.strftime('%Y'),
                           'Q':self.data['origin'].dt.year.astype(str) +'-'+ 'Q' + self.data['origin'].dt.quarter.astype(str),
                           'M':self.data['origin'].dt.strftime('%Y-%b')}
            tri = pd.pivot_table(dev_lag, values='values', index='origin',
                                 columns='dev_lag', aggfunc="sum").sort_index()
            tri.columns = [item for item in tri.columns]
            #fills in lower diagonal with NANs
            tri = self.__fill_inner_diagonal_nans(tri)
            if inplace == True:
                self.data = tri
                self.dataform = 'triangle'
                return self
            if inplace == False:
                new_instance = copy.deepcopy(self)
                return new_instance.data_as_triangle(inplace=True)
        else:
            return self

    def data_as_table(self, inplace=False):
        """Method to convert triangle form to tabular form.

        Arguments:
            inplace: bool
                Set to True will update the instance data attribute inplace

        Returns:
            Updated instance `data` parameter if inplace is set to True otherwise it returns a pandas.DataFrame
        """

        if inplace == True:
            if self.dataform == 'triangle':
                idx = list(self.data.index.names)
                self.data = pd.melt(self.data.reset_index(),id_vars=idx, value_vars=self.data.columns,
                               var_name='dev_lag', value_name='values').dropna().set_index(idx).sort_index().reset_index()
                self.data = self.data.merge(self.__get_development(), how='inner', on=['origin','dev_lag']).drop('dev_lag',axis=1).sort_values(['origin','development'])
            self.dataform = 'tabular'
            return self
        if inplace == False:
            new_instance = copy.deepcopy(self)
            return new_instance.data_as_table(inplace=True)
        else:
            return self

    def __fill_inner_diagonal_nans(self, data):
        '''purpose of this method is to provide data clean-up with missing Values
        in the middle of our data.
        '''
        cols = data.columns
        index = data.index
        temp = np.array(data)
        key = []
        temp_array = np.nan_to_num(np.array(data))
        for i in range(len(cols)):
            x = np.where(np.isfinite(temp[:,i][::-1]))[0][0]
            key.append(x)
            if x > 0:
                temp_array[:,i][-x:] = np.nan
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
            if self.dataform != 'triangle':
                self.data_as_triangle(inplace=True)
                self.cum_to_incr(inplace=True)
                self.data_as_table(inplace=True)
                return self
            a = np.array(self.data)
            incr = pd.DataFrame(np.concatenate((a[:,0].reshape(a.shape[0],1),a[:,1:]-a[:,:-1]),axis=1))
            incr.index = self.data.index
            incr.columns = self.data.columns
            self.data = incr
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
            if self.dataform != 'triangle':
                self.data_as_triangle(inplace=True)
                self.incr_to_cum(inplace=True)
                self.data_as_table(inplace=True)
                return self
            self.data = self.data.cumsum(axis=1)
            return self
        if inplace == False:
            new_instance = copy.deepcopy(self)
            return new_instance.incr_to_cum(inplace=True)

    def get_latest_diagonal(self):
        if self.dataform == 'tabular':
            x = self.data_as_triangle().data
        else:
            x = self.data
        return pd.DataFrame({'dev_lag':[x.columns[np.where(np.logical_not(np.isnan(row)))][-1] for index,row in x.iterrows()],
                             'values':[row.iloc[np.where(np.logical_not(np.isnan(np.array(row))))[0][-1]] for index, row in x.iterrows()]},
                index=x.index)

    def __add__(self, obj):
        if isinstance(obj, Triangle):
            y = copy.deepcopy(self)
            x = copy.deepcopy(obj)
            if x.dataform == 'triangle':
                x.data_as_table(inplace=True)
            if y.dataform == 'triangle':
                y.data_as_table(inplace=True)

            y.data['values']=y.data['values']+x.data['values']
            if self.dataform == 'triangle':
                y.data_as_triangle(inplace=True)
            return y
        else:
            return NotImplemented

    def __radd__(self, obj):
        if obj == 0:
            return self
        else:
            return self.__add__(obj)

    def __sub__(self, obj):
        if isinstance(obj, Triangle):
            y = copy.deepcopy(self)
            x = copy.deepcopy(obj)
            if x.dataform == 'triangle':
                x.data_as_table(inplace=True)
            if y.dataform == 'triangle':
                y.data_as_table(inplace=True)

            y.data['values']=y.data['values']-x.data['values']
            if self.dataform == 'triangle':
                y.data_as_triangle(inplace=True)
            return y
        else:
            return NotImplemented

    def __mul__(self, obj):
        if isinstance(obj, Triangle):
            y = copy.deepcopy(self)
            x = copy.deepcopy(obj)
            if x.dataform == 'triangle':
                x.data_as_table(inplace=True)
            if y.dataform == 'triangle':
                y.data_as_table(inplace=True)

            y.data['values']=y.data['values']*x.data['values']
            if self.dataform == 'triangle':
                y.data_as_triangle(inplace=True)
            return y
        else:
            return NotImplemented

    def __truediv__(self, obj):
        if isinstance(obj, Triangle):
            y = copy.deepcopy(self)
            x = copy.deepcopy(obj)
            if x.dataform == 'triangle':
                x.data_as_table(inplace=True)
            if y.dataform == 'triangle':
                y.data_as_table(inplace=True)

            y.data['values']=y.data['values']/x.data['values']
            if self.dataform == 'triangle':
                y.data_as_triangle(inplace=True)
            return y
        else:
            return NotImplemented

    def grain(self, grain='', incremental=False, inplace=False):
        if inplace == True:
            if grain == '':
                grain = 'O' + self.origin_grain + 'D' + self.development_grain
            if self.dataform == 'triangle':
                self.data_as_table(inplace=True)
                self.grain(grain=grain, inplace=True)
                self.data_as_triangle(inplace=True)
                return self

            # The real function
            ograin = grain[1:2]
            if self.origin_grain != ograin:
                self.origin_grain = ograin
                temp = self.data
                temp['origin'] = pd.PeriodIndex(temp['origin'],freq=ograin).to_timestamp()
                if incremental == True:
                    self.incr_to_cum(inplace=True)
                self.data = pd.pivot_table(temp,index=['origin','development'],values='values', aggfunc='sum').reset_index()

            self.data_as_triangle(inplace=True)
            dgrain = grain[-1]
            if self.development_grain != dgrain:
                init_lag = self.data.iloc[-1].dropna().index[-1]
                final_lag = self.data.iloc[0].dropna().index[-1]
                freq = {'M':{'Y':12,'Q':3,'M':1},
                        'Q':{'Y':4,'Q':1}}
                self.data[[item for item in range(init_lag,final_lag,freq[self.development_grain][dgrain])]]
                self.data_as_table(inplace=True)
                if incremental == True:
                    self.cum_to_incr(inplace=True)
                self.development_grain = dgrain
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
