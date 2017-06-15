# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import os.path


path = os.path.abspath('')[:-6] + 'data\\'
GenInsLong = pd.read_pickle(path + 'GenInsLong')

 
class triangle:
    origin_names = ['accyr', 'accyear', 'accident year', 'origin', 'accmo', 'accpd', 'accident month']
    dev_names = ['devpd', 'dev', 'development month', 'devyr', 'devyear']
     
    def __init__(self, data=None, origin = None, dev = None, values = 'values', dataform = None):
        # Profile the dataset
        dev_in_col_bool = False
        dev_in_index_bool = False
        dev_in_index_T_bool = False
        origin_in_col_bool = False
        origin_in_index_bool = False
        origin_in_index_T_bool = False
        
        
        # Currently only support pandas dataframes as data
        if str(type(data)) != '<class \'pandas.core.frame.DataFrame\'>':
            print('Data is not a proper triangle')
            # Need to figure out how to destroy the object on init fail
            return
        ##### Identify Origin Profile ####
        if origin == None:
            self.origin = origin  
            origin_match = [i for i in origin_names if i in data.columns]
            if len(origin_match)==1:
                self.origin = origin_match[0]
                origin_in_col_bool = True
            if len(origin_match)==0:
                # Checks for common origin names in dataframe index
                origin_match = [i for i in origin_names if i in data.index.name]
                if len(origin_match)==1:
                    self.origin = origin_match[0]
                    origin_in_index_bool = True
        else:
            self.origin = origin  
            
            
        ##### Identify dev Profile ####   
        if dev == None:
            self.dev = dev
            dev_match = [i for i in dev_names if i in data.columns]
            if len(dev_match)==1:
                self.dev = dev_match[0]
                dev_in_col_bool = True
        else:
            self.dev = dev
                
        self.data = data
        
        self.dataform = dataform
        if dev_in_col_bool == True and origin_in_col_bool == True:
            self.dataform = 'tabular'
        
        
        self.values = values
          
    def dataAsTable(self, inplace=False):
        # will need to create triangle class that has origin and dev
        lx = pd.DataFrame()
        if self.dataform == 'triangle':
            for val in range(len(self.data.T.index)):
                df = pd.DataFrame(self.data.iloc[:,val].rename('value'))
                df['dev']= int(self.data.iloc[:,val].name)
                lx = lx.append(df)
            lx.dropna(inplace=True)
            if inplace == True:
                self.data= lx[['dev','value']]
                self.form = 'tabular'
        return lx[['dev','value']]
        

    def dataAsTriangle(self, inplace=False):
        if self.dataform == 'tabular':
            triangle = pd.pivot_table(self.data,values=self.values,index=self.origin, columns=self.dev)
            triangle.columns = [str(item) for item in triangle.columns]
            if inplace == True:
                self.data = triangle   
                self.form = 'triangle'
        return triangle
        
    def incr2cum(self, inplace=False):
        incr = pd.DataFrame(self.data.iloc[:,0])
        for val in range(1, len(self.data.T.index)):
            incr = pd.concat([incr,self.data.iloc[:,val]+incr.iloc[:,-1]],axis=1)
        incr = incr.rename_axis('dev', axis='columns')
        incr.columns = triangle.T.index
        if inplace == True:
            self.data = incr
        return incr     
    
    def cum2incr(self, inplace=False):
        incr = self.data.iloc[:,0]
        for val in range(1, len(self.data.T.index)):
            incr = pd.concat([incr,self.data.iloc[:,val]-self.data.iloc[:,val-1]],axis=1)
        incr = incr.rename_axis('dev', axis='columns')
        incr.columns = self.data.T.index
        if inplace == True:
            self.data = incr        
        return incr   

  

    

    