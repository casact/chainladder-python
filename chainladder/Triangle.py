﻿# -*- coding: utf-8 -*-
"""
The Triangle Module includes the Triangle class.  This base structure of the
class is a pandas.DataFrame.  
"""

from pandas import DataFrame, concat, pivot_table, Series, DatetimeIndex, to_datetime, to_timedelta, melt, read_json, to_numeric
from pandas.tseries.offsets import MonthEnd, QuarterEnd, YearEnd
import numpy as np
import functools
from chainladder.UtilityFunctions import Plot
from bokeh.palettes import Spectral10
import re
import copy
import json

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
        dev : numpy.array or pandas.Series
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
        origin : numpy.array or pandas.Series
            Refer to parameter value.
        dev : numpy.array or pandas.Series
            Refer to parameter value.
        values : str
            Refer to parameter value.
        dataform : str
            Refer to parameter value.

    """

    def __init__(self, data=None, origin=None, development=None, values=None, dataform=None):
        # Currently only support pandas dataframes as data            
        if type(data) is not DataFrame:
            raise TypeError(str(type(data)) + ' is not a valid datatype for the Triangle class.  Currently only DataFrames are supported.')
        if dataform == None:
            if data.shape[1]>= data.shape[0]:
                dataform = 'triangle'
            else:
                dataform = 'tabular'
        self.data = copy.deepcopy(data)
        if development is None and dataform == 'tabular':
                # If development is missing, it is likely an exposure triangle, and needs to conform
                development = 'index'
                temp = self.data.pivot_table(values=values, index=origin, aggfunc='sum').sort_index(ascending=False).reset_index().reset_index().set_index(origin).sort_index()
                temp['index']=temp['index']+1
                self.data = self.data.merge(temp['index'].reset_index(), how='inner', on=origin)
                
        self.data.columns = [str(item) for item in self.data.columns]
        self.dataform = dataform
        if dataform == 'triangle':
            #self.data['origin'] = self.data.index.names # does not work for multi-index
            self.origin = list(self.data.index.names)
            self.data.columns = list(np.array(self.data.columns).astype(np.int64))
            self.data.reset_index(inplace=True)
            self.values='values'
            self.dev_lag = 'dev_lag'
            self.origin_dict = self.__set_period('origin') 
            self.ncol = len(data.columns)
            self.__minimize_data()         
        else:
            self.origin = origin
            self.origin_dict = self.__set_period('origin')
            self.development = development
            self.dev_dict = self.__set_period('dev')
            self.dev_lag = 'dev_lag'
            self.values = 'values'
            self.data['values'] = self.data[values]
            self.ncol = len(self.data['dev_lag'].drop_duplicates())
            self.__minimize_data()
            self.data.set_index([self.origin_dict[keys] for keys in self.origin_dict],inplace=True)
            self.data.sort_index(inplace=True)
        self.__get_lag_granularity()
           
    def __set_period(self, per_type):
        """ This method will take an variable length array of columns and convert them to
        separate year/quarter/month columns if this information is extractable.
        It also defines the development lag at the most granular level.
        """
        if per_type == 'origin':
            period = self.origin
        else:
            period = self.development
        period_dict = {}
        if type(period) == str:
            period = [period]
        if len(period) > 1:
            for item in period:
                #Runs through many column names to determine if they measure year, quarter or month
                #Items named YearQuarter, yearmonth or some combo would fail, but I generally don't expect
                #this when multiple separate period columns are provided.
                if len(re.findall(r'y[a-z]*r|y|Y[A-Z]*R|Y',item))>0:
                    period_dict['year'] = item
                if len(re.findall(r'q[a-z]*r|q|Q[A-Z]*R|Q',item))>0 and len(self.data[item].unique()) == 4:
                    period_dict['quarter'] = item
                if len(re.findall(r'm[a-z]*|M[A-Z]*',item))>0 and len(self.data[item].unique()) == 12:
                    period_dict['month'] = item
            if 'month' in period_dict.keys():
                colname = str(period_dict['year'])+'-'+str(period_dict['month'])
                self.data[colname]=self.data[period_dict['year']].astype(str) + '-' + self.data[period_dict['month']].astype(str)
            if 'month' not in period_dict.keys() and 'quarter' in period_dict.keys():
                colname = str(period_dict['year'])+'-'+str(period_dict['quarter'])
                self.data[colname]=self.data[period_dict['year']].astype(str)+'Q'+self.data[period_dict['quarter']].astype(str)
            dates = DatetimeIndex(to_datetime(self.data[colname]))
            self.data.drop([colname], axis=1, inplace=True) # get rid of temporary concat column
            if per_type == 'origin':self.data = self.data.drop(self.origin, axis=1) # get rid of original columns
            
        if len(period) == 1:
            dates = DatetimeIndex(to_datetime(self.data[period[0]]))
        period_dict['year'] = per_type + '_year'
        if len(set(dates.quarter)) > 1:
            self.data[per_type + '_year']=dates.year
        if len(set(dates.quarter)) == 4:
            period_dict['quarter'] = per_type + '_quarter'
            self.data[per_type + '_quarter']=dates.quarter
        if len(set(dates.month)) == 12:
            period_dict['month'] = per_type + '_month'
            self.data[per_type + '_month']=dates.month
            
        if per_type == 'dev':
            # sets the most granular dev_lag available.
            if 'month'in period_dict.keys():
                self.dev_grain = 'month'
            elif 'quarter' in period_dict.keys() and 'month' not in period_dict.keys():
                self.dev_grain = 'quarter'
            else:
                self.dev_grain = 'year'
                
            if len(set(dates.year)) == 1:
                #Implies that a dev lag has been specified over dev period
                if np.all((self.data[period[0]].unique()<3000) & (self.data[period[0]].unique()>1900)): # carve out unique case of YYYY.
                    self.data[per_type + '_year']=self.data[period[0]]
                    self.data['dev_lag']=self.data['dev_year']-self.data['origin_year']+1
                else:
                    self.data['dev_lag'] = self.data[period]
            else:
                # set development lag, this should handle both symmetric and asymmentric triangles
                if len(set(dates.month)) == 12: # use month
                    self.data['dev_lag']=(self.data['dev_year']-self.data['origin_year'])*12 + self.data['dev_month']-self.data['origin_month']+1
                elif len(set(dates.quarter)) == 4:
                    self.data['dev_lag']=(self.data['dev_year']-self.data['origin_year'])*4 + self.data['dev_quarter']-self.data['origin_quarter']+1
                else:
                    self.data['dev_lag']=self.data['dev_year']-self.data['origin_year']
                    
        if per_type == 'origin' and len(set(dates.year)) == 1:
            self.data['origin_year'] = self.data[period]              
            if period != ['origin_year']:
                self.data = self.data.drop(self.origin, axis=1)
        
        return period_dict     
    
    def __minimize_data(self):
        if self.dataform == 'triangle':
            self.data = self.data.set_index([self.origin_dict[key] for key in self.origin_dict])
        if self.dataform == 'tabular':
            cols = [self.origin_dict[key] for key in self.origin_dict] + ['dev_lag', 'values']
            self.data = self.data[cols] 
        
    def __get_lag_granularity(self):
        temp = self.data_as_triangle()
        yval = (len(temp.data.iloc[-1]) - temp.data.iloc[-1] .count()) - (len(temp.data.iloc[-2]) - temp.data.iloc[-2] .count())
        xval = len(temp.origin_dict.keys())
        if yval == 4:
            self.dev_grain = 'quarter'
        if yval in (3,12):
            self.dev_grain = 'month'
        if yval == 1:
            temp_dict={1:'year',2:'quarter',3:'month'}
            self.dev_grain =  temp_dict[xval]
            
    def __repr__(self):   
        return str(self.data)
    
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
                self.data = melt(self.data.reset_index(),id_vars=idx, value_vars=self.data.columns, 
                               var_name='dev_lag', value_name='values').dropna().set_index(idx).sort_index()
            self.dataform = 'tabular'
            return self
        if inplace == False:
            new_instance = copy.deepcopy(self)
            return new_instance.data_as_table(inplace=True)
        else:
            return self

    def data_as_triangle(self, inplace=False):
        """Method to convert tabular form to triangle form.
    
        Arguments:
            inplace: bool
                Set to True will update the instance data attribute inplace 
    
        Returns:
            Updated instance `data` parameter if inplace is set to True otherwise it returns a pandas.DataFrame
        """
        if self.dataform == 'tabular':
            available_origin_periods = ['origin_year', 'origin_quarter','origin_month']
            my_origin_periods = [item for item in available_origin_periods if item in self.data.reset_index().columns]
            tri = pivot_table(self.data, values=self.values, index=
                              my_origin_periods, columns=self.dev_lag, aggfunc="sum", fill_value=0).sort_index()
            tri.columns = [item for item in tri.columns]
            # temp fills in lower diagonal with NANs, only works for symmetric triangles
            if tri.shape[0] == tri.shape[1]:
                temp = (np.flip(np.tril(np.empty(tri.shape)*np.NAN, k=-1),axis=1)+1)
                tri = tri * temp
            else:
                tri = pivot_table(self.data, values=self.values, index=
                              my_origin_periods, columns=self.dev_lag, aggfunc="sum", fill_value=0).sort_index()
            if inplace == True:
                self.data = tri
                self.dataform = 'triangle'
                return self
            if inplace == False:
                new_instance = copy.deepcopy(self)
                return new_instance.data_as_triangle(inplace=True)
        else:
            return self


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
            self.data = self.data.T.cumsum().T
            return self
        if inplace == False:
            new_instance = copy.deepcopy(self)
            return new_instance.incr_to_cum(inplace=True)
        
    
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
            incr = DataFrame(np.concatenate((a[:,0].reshape(a.shape[0],1),a[:,1:]-a[:,:-1]),axis=1))
            incr.index = self.data.index
            incr.columns = self.data.columns
            self.data = incr
            return self
        if inplace == False:
            new_instance = copy.deepcopy(self)
            return new_instance.cum_to_incr(inplace=True)
    
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
        # need tabular view
        data = self.data_as_table().data.reset_index()
        if 'month' in self.origin_dict.keys():
            data['origin_period'] = to_datetime(data[self.origin_dict['year']].astype(str) + '-' + (data[self.origin_dict['month']]).astype(str))
        elif 'quarter' in self.origin_dict.keys() and 'month' not in self.origin_dict.keys():
            data['origin_period'] = to_datetime(data[self.origin_dict['year']].astype(str) + 'Q' + (data[self.origin_dict['quarter']]).astype(str))
        else:
            data['origin_period'] = to_datetime(data[self.origin_dict['year']].astype(str))
        data = data.set_index([self.origin_dict[key] for key in self.origin_dict])
        params = {'month':[1,'M'],'quarter':[3,'M'],'year':[1,'Y']}[self.dev_grain]
        data['cal_period'] = (data['origin_period']+to_timedelta(((data['dev_lag'])*params[0]).astype(int),unit=params[1])).dt.date
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
        group_subset = np.concatenate([self.tri_dict[item] for item in query_list],axis=0)
        value_subset = np.sum([group_subset[:,idx] for idx in [self.columns.index(item) for item in [self.values[i] for i in query_values]]],axis=0)
        val_col = '__values' if 'values' in self.columns else 'values'
        df_subset = DataFrame(np.concatenate([group_subset, np.expand_dims(value_subset,axis=1)],axis=1), columns=self.columns+[val_col])
        df_subset['values'] = to_numeric(df_subset['values'])
        if len(query_values) == 2 and operator!="+":
            print(self.values[query_values[1]])
            tri1 = Triangle(df_subset, origin=self.origin, development=self.development, values=self.values[query_values[0]])
            tri2 = Triangle(df_subset, origin=self.origin, development=self.development, values=self.values[query_values[1]])
            print(tri1.data_as_triangle()/tri2.data_as_triangle())
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

