# -*- coding: utf-8 -*-
"""
The Triangle Module includes the Triangle class.  This base structure of the
class is a pandas.DataFrame.
"""

from pandas import (DataFrame, concat, pivot_table, Series, DatetimeIndex,
                    to_datetime, to_timedelta, melt, read_json, to_numeric,
                    PeriodIndex)
from pandas.tseries.offsets import MonthEnd, QuarterEnd, YearEnd
import numpy as np
import functools
from chainladder.UtilityFunctions import Plot
from bokeh.palettes import Spectral10
import re
import json
import warnings


class Triangle():
    """Triangle class is the basic data representation of an actuarial triangle.

    Historical insurance data is often presented in form of a triangle
    structure, showing the development of claims over time for each exposure
    (origin) period. An origin period could be the year the policy was written
    or earned, the loss occurrence period, or the date the loss was reported.
    Of course the origin period doesn’t have to be yearly, e.g. quarterly,
    monthly, or semi-annual origin periods are also often used. The development
    period of an origin period is also called age or lag. Data on the diagonals
    present payments in the same calendar period.

    Note, data of individual policies is usually aggregated to homogeneous
    lines of business, division levels or perils. Most reserving methods of the
    ChainLadderpackage expect triangles as input data sets with development
    periods along the columns and the origin period in rows. The package comes
    with several example triangles.

    `Proper Citation Needed... <https://github.com/mages/ChainLadder>`_

    Parameters:
        data : pandas.DataFrame
            A DataFrame representing the triangle. When in triangle form, the
            index of the dataframe should be the origin period. The columns
            names will be the development periods.
        dataform : str
            A string value that takes on one of two values
            ['triangle' and 'tabular']
        cumulative : boolean
            When True (the default) values in the data should reflect cumulative
            values. Otherwise, values reflect incremental values.
        origin : str
            Required or 'tabular' format, should be the column representing the
            origin periods.
        development : str
            Required for 'tabular' format, should be the column representing
            the  development periods.

    Attributes:
        data : pandas.DataFrame
            A DataFrame representing the triangle
        origin : numpy.array or pandas.Series
            Refer to parameter value.
        dev : numpy.array or pandas.Series
            Refer to parameter value.
        values : str
            Refer to parameter value.
        dataform : str
            Refer to parameter value.

    """
    def __init__(self, data=None, dataform='triangle', cumulative=True,
                 origin=None, development=None):
        # Currently only support pandas dataframes as data
        if (type(data) is not DataFrame) and (type(data) is not Series):
            raise TypeError(str(type(data)) + ' is not a valid datatype for '
                            'the Triangle class.  Currently only DataFrames '
                            'and Series are supported.')
        if (type(data) is Series) and (dataform == 'triangle'):
            raise TypeError('Series may only be used as a tabular dataform')
        data = data.copy()
        # Its simplest to force the data into a standard format... we'll use
        # tabular form out of convenience

        if dataform == 'triangle':
            data = data.stack()
        elif type(data) is DataFrame:
            data = data.set_index([origin, development])

        self.data = data
        self.dataform = 'tabular'
        self.cumulative = cumulative

        self.ncol = len(self.data.index.levels[1])

    def __repr__(self):
        return str(self.data)

    def __add__(self, other):
        if isinstance(other, Triangle):
            output = self.data + other.data
            return Triangle(output, dataform='tabular')
        else:
            return NotImplemented

    def __radd__(self, other):
        if other is None:
            return self
        else:
            return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Triangle):
            output = self.data - other.data
            return Triangle(output, dataform='tabular')
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Triangle):
            output = self.data * other.data
            return Triangle(output, dataform='tabular')
        else:
            return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, Triangle):
            output = self.data / other.data
            return Triangle(output, dataform='tabular')
        else:
            return NotImplemented

    def data_as_table(self):
        """Method to convert triangle form to tabular form.

        Returns:
            Pandas dataframe of Triangle in trabular form.
        """
        return self.data

    def data_as_triangle(self, inplace=False):
        """Method to convert tabular form to triangle form.

        Returns:
            Returns a pandas.DataFrame of the data in triangle form
        """
        return self.data.unstack()

    def incr_to_cum(self, inplace=True):
        """Method to convert an incremental triangle into a cumulative triangle.  Note,
        the triangle must be in triangle form.

        Arguments:
            inplace: bool
                Set to True will update the instance data attribute inplace

        Returns:
            Updated instance `data` parameter if inplace is set to True.
            Otherwise it returns a pandas.DataFrame

        """
        if not self.cumulative:
            data = self.data.copy()
            data = data.groupby(level=[0]).cumsum()
            if inplace:
                self.data = data
                self.cumulative = True
                return
            else:
                return data
        else:
            if inplace:
                return
            else:
                return self.data

    def cum_to_incr(self, inplace=True):
        """Method to convert an cumulative triangle into a incremental
        triangle. Note, the triangle must be in triangle form.

        Arguments:
            inplace: bool
                Set to True will update the instance data attribute inplace

        Returns:
            Updated instance `data` parameter if inplace is set to True.
            Otherwise it returns a pandas.DataFrame

        """
        if self.cumulative:
            data = self.data.copy()
            for origin in data.index.levels[0]:
                incr = data.loc[origin] - data.loc[origin].shift(1).fillna(0)
                data.loc[origin] = incr.values
            if inplace:
                self.data = data
                self.cumulative = False
                return
            else:
                return data
        else:
            if inplace:
                return
            else:
                return self.data

    def get_latest_diagonal(self):
        #return DataFrame({'dev_lag':[self.data.columns[len(row.dropna())-1] for index,row in self.data.iterrows()],
        #                  'values':[row.dropna().iloc[-1] for index, row in self.data.iterrows()]},
        #        index=self.data.index)
        if self.dataform == 'tabular':
            x = self.data_as_triangle().data
        else:
            x = self.data
        return DataFrame({'dev_lag':[x.columns[np.where(np.logical_not(np.isnan(row)))][-1] for index,row in x.iterrows()],
                          'values':[row.iloc[np.where(np.logical_not(np.isnan(np.array(row))))[0][-1]] for index, row in x.iterrows()]},
                index=x.index)

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
        xvals = [i+1 for i in range(len(self.data.columns))]
        plot_dict = {'triangle':{'Title':'Latest Triangle Data',
                                     'XLabel':'Development Period',
                                     'YLabel':'Values',
                                     'chart_type_dict':{'type':['line'],
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
                                                       }}
                    }
        return plot_dict


    def slide_right(self, right=True, inplace=False):
        """
        Function used for making aggregations across origin period.
        """

        if inplace == True:
            if self.dataform=='tabular':
                self.data_as_triangle(inplace=True)
                self.slide_right(right=right, inplace=True)
                self.data_as_table(inplace=True)
                return self
            my_arr = np.array(self.data)
            if right==False:
                for i in range(len(my_arr)):
                    my_arr[i] = np.roll(my_arr[i],np.sum((np.isnan(my_arr[i])-1)*-1))
            else:
                for i in range(len(my_arr)):
                    my_arr[i] = np.roll(my_arr[i],np.sum(np.isnan(my_arr[i])))
            self.data = DataFrame(my_arr,index=self.data.index,columns=self.data.columns)
            return self
        if inplace == False:
            new_instance = copy.deepcopy(self)
            return new_instance.slide_right(right=right, inplace=True)


    def grain(self, grain='', incremental=False, inplace=False):
        # data needs to be in triangle form to use this.
        if grain == '':
            grain = self.get_grain()
        if inplace == True:
            if self.dataform == 'tabular':
                self.data_as_triangle(inplace=True)
                self.grain(grain=grain, inplace=True)
                self.data_as_table(inplace=True)
                return self
            ograin = grain[:2]
            if ograin == 'OQ':
                idx = [self.origin_dict['year'], self.origin_dict['quarter']]
                if 'month' in self.origin_dict: del self.origin_dict['month']
            elif ograin == 'OY':
                idx = [self.origin_dict['year']]
                if 'month' in self.origin_dict: del self.origin_dict['month']
                if 'quarter' in self.origin_dict: del self.origin_dict['quarter']
            elif ograin == 'OM':
                idx = [self.origin_dict['year'], self.origin_dict['month']]
            else:
                idx = self.data.index
            if incremental == True:
                self.incr_to_cum(inplace=True)

            self.data = pivot_table(self.slide_right().data_as_table().data,index=idx,columns=self.dev_lag,values=self.values, aggfunc='sum')
            self.slide_right(False,True)

            dgrain = grain[-2:]
            initial_lag = self.data.iloc[-1].count().item()
            dev_grain = self.dev_grain
            cols = self.data.columns.astype(int)
            if dgrain == 'DQ':
                step = {'quarter':1, 'month':3}[dev_grain]
            elif dgrain == 'DY':
                step = {'year':1, 'quarter':4, 'month':12}[dev_grain]
            else:
                step = 1
            self.dev_grain = dev_grain
            new_dev_lags = list(set([item+initial_lag for item in range(0-step*10,max(cols),step)]).intersection(cols))
            new_dev_lags.sort()
            self.data = self.data[new_dev_lags]
            if incremental == True:
                self.cum_to_incr(inplace=True)
            self.ncol = len(self.data.columns)
            return self
        if inplace == False:
            new_instance = copy.deepcopy(self)
            return new_instance.grain(grain=grain, incremental=incremental, inplace=True)

    def lag_to_date(self):
        ''' Converts development lag to a date.  Needs work as it doesn't work on leap year
        '''
        data = self.data_as_table().data.reset_index()
        if 'month' in self.origin_dict.keys():
            data['origin_period'] = to_datetime(data[self.origin_dict['year']].astype(str) + '-' + (data[self.origin_dict['month']]).astype(str))
        elif 'quarter' in self.origin_dict.keys() and 'month' not in self.origin_dict.keys():
            data['origin_period'] = to_datetime(data[self.origin_dict['year']].astype(str) + 'Q' + (data[self.origin_dict['quarter']]).astype(str))
        else:
            data['origin_period'] = to_datetime(data[self.origin_dict['year']].astype(str))
        data = data.set_index([self.origin_dict[key] for key in self.origin_dict])
        params = {'month':[1,'M'],'quarter':[3,'M'],'year':[1,'Y']}[self.dev_grain]
        data['cal_period'] = (data['origin_period']+to_timedelta(((data['dev_lag'].astype(int))*params[0]).astype(int),unit=params[1])).dt.date
        data = data[['dev_lag','cal_period']]
        return data

    def get_grain(self):
        if 'month' in self.origin_dict.keys():
            temp = 'OM'
        elif 'quarter' in self.origin_dict.keys():
            temp = 'OQ'
        else:
            temp = 'OY'
        temp = temp + {'year':'DY','quarter':'DQ','month':'DM'}[self.dev_grain]
        return temp

    def fill_inner_diagonal_nans(self, data):
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
        return DataFrame(temp_array, index=index, columns=cols)


# Pandas-style indexing classes to support an array of triangles
#class TriangleSet2:
#    def __init__(self, data=None, origin=None, development=None, values=None, groups=None):
#        self.origin = origin
#        self.development = development
##            self.values = values
 #       if type(values) is str:
 #           self.values = [values]
 #       if groups is None:
 #           self.group_list = ['total']
 #           self.tri_dict= dict([(item, dict([(val, Triangle(data, origin=origin, development=development, values=val)) for val in self.values])) for item in self.group_list])
 #       elif type(groups) is str:
 ##           self.tri_dict= dict([(item, dict([(val, Triangle(data = data[data[groups]==item], origin=origin, development=development, values=val)) for val in self.values])) for item in self.group_list])
 #       else:
 #           def conjunction(*conditions):
 #               return functools.reduce(np.logical_and, conditions)
 #           self.group_list = data.groupby(groups).size().reset_index().drop(0, axis=1)
 #           self.tri_dict = dict([(idx, dict([(val, Triangle(data = dataset, origin=origin, development=development, values=val)) for val in self.values])) for idx, dataset in enumerate([data[conjunction(*[data[k]==v for k,v in dict_item.items()])] for dict_item in [dict(self.group_list.iloc[row]) for row in range(len(self.group_list))]])])
 #       self.iloc = iloc(self.tri_dict)
 #       self.loc = loc(self.tri_dict)
 #       self.shape = (len(self.tri_dict), len(self.tri_dict[list(self.tri_dict.keys())[0]]))

 #   def __len__(self):
 #       return len(self.tri_dict)

class TriangleSet:
    '''Class that allows for the management of mutliple triangles in one object.
    This is useful when you have a single dataset has multiple values (paid, incurred, etc), or
    multiple reserve groups (property, liability).  It will create a set of triangles
    that can be accessed by their index numbers.
    '''
    def __init__(self, data=None, origin=None, development=None, values=None, groups=None):
        self.origin = origin
        self.development = development
        if type(values) is list:
            self.values = values
        if type(values) is str:
            self.values = [values]
        if groups is None or groups == [] or groups == '':
            groups = ['total']
            data['total'] = 'total'
        self.group_list = data.groupby(groups).size().reset_index().drop(0, axis=1)
        self.columns = list(data.columns)
        self.data = np.array(data)
        self.tri_dict = dict([(iloc, self.data[np.product([np.where(self.data[:,item]==self.group_list.iloc[iloc,num],True,False) for num,item in enumerate([self.columns.index(col) for col in list(self.group_list.columns)])],axis=0, dtype=bool)]) for iloc in range(len(self.group_list))])
        #self.iloc = iloc(self.tri_dict)
        #self.loc = loc(self.tri_dict)
        self.shape = (len(self.tri_dict), len(self.tri_dict[list(self.tri_dict.keys())[0]]))

    def __len__(self):
        return len(self.tri_dict)

    def get_triangles(self, query_values, query_list, operator='+'):
        '''function to access individual triangles by index number.
        '''
        group_subset = np.concatenate([self.tri_dict[item] for item in query_list],axis=0)
        value_subset = np.sum([group_subset[:,idx] for idx in [self.columns.index(item) for item in [self.values[i] for i in query_values]]],axis=0)
        val_col = '__values' if 'values' in self.columns else 'values'
        df_subset = DataFrame(np.concatenate([group_subset, np.expand_dims(value_subset,axis=1)],axis=1), columns=self.columns+[val_col])
        df_subset['values'] = to_numeric(df_subset['values'])
        if len(query_values) == 2 and operator!="+":
            tri1 = Triangle(df_subset, origin=self.origin, development=self.development, values=self.values[query_values[0]])
            tri2 = Triangle(df_subset, origin=self.origin, development=self.development, values=self.values[query_values[1]])
            if operator in ["-"]:
                return tri1-tri2
            if operator in ["×","*"]:
                return tri1*tri2
            if operator in ["÷","/"]:
                return (tri1.data_as_triangle()/tri2.data_as_triangle()).data_as_table()
        else:
            return Triangle(df_subset, origin=self.origin, development=self.development, values='values')





#class iloc:
#    def __init__(self, tri_dict):
#        self.tri_dict = tri_dict

 #   def __getitem__(self, idx):
        # slicing

 #       if type(idx[0]) is int:
 #           row = slice(idx[0], idx[0]+1,1)
 #       else:
 #           row = idx[0]
 #       if type(idx[1]) is int:
 #           col = slice(idx[1], idx[1]+1,1)
 #       else:
 #           col = idx[1]
 #       group_start, group_stop, group_step = self.parse_slice(row)
 #       val_start, val_stop, val_step = self.parse_slice(col)
 #       temp = [item[1] for item in list(self.tri_dict.items())[group_start:group_stop:group_step]]
 #       temp2 = [[item[1] for item in list(tempdict.items())[val_start:val_stop:val_step]] for tempdict in temp]
 #       temp3 = [item for sublist in temp2 for item in sublist]
 #       if len(temp3) == 1:
 #           return temp3[0]
 #       else:
 #           return temp3

  #  def parse_slice(self, sl):
  #      if sl.start is None:
  #          start = 0
 #       else:
 #           start = sl.start
 #       if sl.stop is None:
 #           stop = len(self.tri_dict.keys())
 #       else:
 #           stop = sl.stop
 #       if sl.step is None:
 #           step = 1
 #       else:
 #           step = sl.step
 #       return start, stop, step


#class loc:
#    def __init__(self, tri_dict):
#        self.tri_dict = tri_dict

#    def __getitem__(self,idx):
#        return self.tri_dict[idx[0]][idx[1]]


#groups_to_iterate_over = list(group_key.keys())
#return_list = []
#for i in range(len(groups_to_iterate_over)):
#    return_list.append([group_key[groups_to_iterate_over[i]][idx] for idx in test[i]])
#bool_array = np.ones(len(group_list), dtype=bool)
#for i in range(len(groups_to_iterate_over)):
#    bool_array = bool_array * np.array(group_list.iloc[:,i].isin(return_list[i]), dtype=bool)
#final = list(group_list[bool_array].index)


#data = np.array(TOT1997)
#columns = list(TOT1997.columns)
#group_list = TS.group_list
#values = TS.values
#origin=TS.origin
#development=TS.development


# Get index of each of the group list columns in master array
#iloc = 0
#groups_to_loop_over = [columns.index(col) for col in list(group_list.columns)]

#query_list = [1,5,3,6,7,84,8,12,64]
#query_values=[3,4,5]
#group_subset = np.concatenate([tri_dict[item] for item in query_list],axis=0)
#value_subset = np.sum([sub_set[:,idx] for idx in [columns.index(item) for item in [values[i] for i in query_values]]],axis=0)
#df_subset = pd.DataFrame(np.concatenate([sub_set, np.expand_dims(new_val,axis=1)],axis=1), columns=columns+['values'])
#df_subset['values'] = pd.to_numeric(df_subset['values'])
#cl.Triangle(df_subset, origin=origin, development=development, values='values').data_as_triangle()
