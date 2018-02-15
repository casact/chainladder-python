﻿# -*- coding: utf-8 -*-
"""
The Triangle Module includes the Triangle class.  This base structure of the
class is a pandas.DataFrame.  
"""

from pandas import DataFrame, concat, pivot_table, Series
from chainladder.UtilityFunctions import Plot
from bokeh.palettes import Spectral10

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
        origin : numpy.array or pandas.Series
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

    def __init__(self, data=None, origin=None, dev=None, values=None, dataform='triangle'):
        # Currently only support pandas dataframes as data
        if type(data) is not DataFrame:
            raise TypeError(str(type(data)) + ' is not a valid datatype for the Triangle class.  Currently only DataFrames are supported.')
        self.data = data
        self.data.columns = [str(item) for item in self.data.columns]
        self.origin = origin
        if origin == None:
            origin_in_col_bool, origin_in_index_bool = self.__set_origin()
        else:
            origin_in_col_bool = True
        self.dev = dev
        if dev == None:
            dev_in_col_bool = self.__set_dev()
        else:
            dev_in_col_bool = True
        self.dataform = dataform
        if dev_in_col_bool == True and origin_in_col_bool == True:
            self.dataform = 'tabular'
        self.values = values
        if values == None and self.dataform == 'tabular':
            self.__set_values()
        if dataform == 'triangle':
            self.ncol = len(data.columns)
        else:
            self.ncol = len(self.data[dev].unique())
        #self.latest_values = Series(
        #    [row.dropna().iloc[-1] for index, row in self.data.iterrows()], index=self.data.index)
        
    def __repr__(self):   
        return str(self.data)
    
    def data_as_table(self, inplace=False):
        """Method to convert triangle form to tabular form.

        Arguments:
            inplace: bool
                Set to True will update the instance data attribute inplace 

        Returns:
            Updated instance `data` parameter if inplace is set to True otherwise it returns a pandas.DataFrame
        """
        lx = DataFrame()
        if self.dataform == 'triangle':
            for val in range(len(self.data.T.index)):
                df = DataFrame(self.data.iloc[:, val].rename('values'))
                df['dev'] = int(self.data.iloc[:, val].name)
                lx = lx.append(df)
            lx.dropna(inplace=True)
            if inplace == True:
                self.data = lx[['dev', 'values']]
                self.dataform = 'tabular'
                self.dev = 'dev'
                self.values = 'values'
            return lx[['dev', 'values']]
        else:
            return self.data

    def data_as_triangle(self, inplace=False):
        """Method to convert tabular form to triangle form.
    
        Arguments:
            inplace: bool
                Set to True will update the instance data attribute inplace 
    
        Returns:
            Updated instance `data` parameter if inplace is set to True otherwise it returns a pandas.DataFrame
        """
        if self.dataform == 'tabular':
            tri = pivot_table(self.data, values=self.values, index=[
                              self.origin], columns=[self.dev]).sort_index()
            tri.columns = [str(item) for item in tri.columns]
            if inplace == True:
                self.data = tri
                self.dataform = 'triangle'
            return tri
        else:
            return self.data


    def incr_to_cum(self, inplace=False):
        """Method to convert an incremental triangle into a cumulative triangle.  Note,
        the triangle must be in triangle form.

        Arguments:
            inplace: bool
                Set to True will update the instance data attribute inplace 

        Returns:
            Updated instance `data` parameter if inplace is set to True otherwise it returns a pandas.DataFrame

        """
        
        incr = self.data.T.cumsum().T
        if inplace == True:
            self.data = incr
        return incr
    def cum_to_incr(self, inplace=False):
        """Method to convert an cumulative triangle into a incremental triangle.  Note,
        the triangle must be in triangle form.

        Arguments:
            inplace: bool
                Set to True will update the instance data attribute inplace 

        Returns:
            Updated instance `data` parameter if inplace is set to True otherwise it returns a pandas.DataFrame

        """
        incr = self.data.iloc[:, 0]
        for val in range(1, len(self.data.T.index)):
            incr = concat([incr, self.data.iloc[:, val] -
                           self.data.iloc[:, val - 1]], axis=1)
        incr = incr.rename_axis('dev', axis='columns')
        incr.columns = self.data.T.index
        if inplace == True:
            self.data = incr
        return incr

    def __set_origin(self):
        """Experimental hidden method. Purpose is to profile the data and autodetect the origin period
        improving the user experience by not requiring the user to supply an explicit origin.

        """
        origin_names = ['accyr', 'accyear', 'accident year', 'origin', 'accmo', 'accpd',
                        'accident month','AccidentYear']
        origin_in_col_bool = False
        origin_in_index_bool = False
        origin_in_index_T_bool = False
        origin_match = [i for i in origin_names if i in self.data.columns]
        if len(origin_match) == 1:
            self.origin = origin_match[0]
            origin_in_col_bool = True
        if len(origin_match) == 0:
            # Checks for common origin names in dataframe index
            origin_match = [
                i for i in origin_names if i in self.data.index.name]
            if len(origin_match) == 1:
                self.origin = origin_match[0]
                origin_in_index_bool = True
        return origin_in_col_bool, origin_in_index_bool

    def __set_dev(self):
        """Experimental hidden method. Purpose is to profile the data and autodetect the dev period
        improving the user experience by not requiring the user to supply an explicit dev.

        """
        ##### Identify dev Profile ####
        dev_names = ['devpd', 'dev', 'development month', 'devyr', 'devyear']
        dev_in_col_bool = False
        dev_in_index_bool = False
        dev_in_index_T_bool = False
        dev_match = [i for i in dev_names if i in self.data.columns]
        if len(dev_match) == 1:
            self.dev = dev_match[0]
            dev_in_col_bool = True
        return dev_in_col_bool

    def __set_values(self):
        """Experimental hidden method. Purpose is to profile the data and autodetect the values parameter
        improving the user experience by not requiring the user to supply an explicit values parameter.
        This is onyl necessary when dataform is 'tabular'.

        """
        ##### Identify values Profile ####
        value_names = ['incurred claims']
        values_match = [i for i in value_names if i in self.data.columns]
        if len(values_match) == 1:
            self.values = values_match[0]
        else:
            self.values = 'values'
        return
    
    def get_latest_diagonal(self):
        return Series([row.dropna(
        ).iloc[-1] for index, row in self.data.iterrows()], index=self.data.index)
    
    def plot(self, ctype='m', plots=['triangle']): 
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
        Plot(ctype, my_dict)
        
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
                                                       'alpha':[1]
                                                       }} 
                    }
        return plot_dict

    