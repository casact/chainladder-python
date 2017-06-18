# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 17:34:23 2017

@author: jboga
"""
import numpy as np
from pandas import DataFrame, concat

class Triangle:
    def __init__(self, data=None, origin = None, dev = None, values = None, dataform = 'triangle'):
        # Currently only support pandas dataframes as data
        if str(type(data)) != '<class \'pandas.core.frame.DataFrame\'>':
            print('Data is not a proper triangle')
            # Need to figure out how to destroy the object on init fail
            return
        self.data = data
        self.origin = origin
        if origin == None:
            origin_in_col_bool, origin_in_index_bool = self.__set_origin()   
        self.dev = dev     
        if dev == None:
            dev_in_col_bool = self.__set_dev() 
        self.dataform = dataform
        if dev_in_col_bool == True and origin_in_col_bool == True:
            self.dataform = 'tabular'
        self.values = values
        if values == None:
            self.__set_values() 
          
    def dataAsTable(self, inplace=False):
        # will need to create triangle class that has origin and dev
        lx = DataFrame()
        if self.dataform == 'triangle':
            for val in range(len(self.data.T.index)):
                df = DataFrame(self.data.iloc[:,val].rename('values'))
                df['dev']= int(self.data.iloc[:,val].name)
                lx = lx.append(df)
            lx.dropna(inplace=True)
            if inplace == True:
                self.data= lx[['dev','values']]
                self.dataform = 'tabular'
                self.dev = 'dev'
                return lx[['dev','values']]
        else:
            return
        

    def dataAsTriangle(self, inplace=False):
        if self.dataform == 'tabular':
            tri = pivot_table(self.data,values=self.values,index=[self.origin], columns=[self.dev]).sort_index()
            tri.columns = [str(item) for item in tri.columns]
            if inplace == True:
                self.data = tri   
                self.dataform = 'triangle'
        return tri
        
    def incr2cum(self, inplace=False):
        incr = DataFrame(self.data.iloc[:,0])
        for val in range(1, len(self.data.T.index)):
            incr = concat([incr,self.data.iloc[:,val]+incr.iloc[:,-1]],axis=1)
        incr = incr.rename_axis('dev', axis='columns')
        incr.columns = self.data.T.index
        if inplace == True:
            self.data = incr
        return incr     
    
    def cum2incr(self, inplace=False):
        incr = self.data.iloc[:,0]
        for val in range(1, len(self.data.T.index)):
            incr = concat([incr,self.data.iloc[:,val]-self.data.iloc[:,val-1]],axis=1)
        incr = incr.rename_axis('dev', axis='columns')
        incr.columns = self.data.T.index
        if inplace == True:
            self.data = incr        
        return incr   
    
    def __set_origin(self):
        ##### Identify Origin Profile ####
        origin_names = ['accyr', 'accyear', 'accident year', 'origin', 'accmo', 'accpd', 
                        'accident month']
        origin_in_col_bool = False
        origin_in_index_bool = False
        origin_in_index_T_bool = False 
        origin_match = [i for i in origin_names if i in self.data.columns]
        if len(origin_match)==1:
            self.origin = origin_match[0]
            origin_in_col_bool = True
        if len(origin_match)==0:
            # Checks for common origin names in dataframe index
            origin_match = [i for i in origin_names if i in self.data.index.name]
            if len(origin_match)==1:
                self.origin = origin_match[0]
                origin_in_index_bool = True
        return origin_in_col_bool, origin_in_index_bool

    def __set_dev(self):
        ##### Identify dev Profile ####
        dev_names = ['devpd', 'dev', 'development month', 'devyr', 'devyear']
        dev_in_col_bool = False
        dev_in_index_bool = False
        dev_in_index_T_bool = False        
        dev_match = [i for i in dev_names if i in self.data.columns]
        if len(dev_match)==1:
            self.dev = dev_match[0]
            dev_in_col_bool = True
        return dev_in_col_bool
    
    def __set_values(self):
        ##### Identify dev Profile ####
        value_names = ['incurred claims'] 
        values_match = [i for i in value_names if i in self.data.columns]
        if len(values_match)==1:
            self.values = values_match[0]
        else:
            self.values = 'values'
        return 
    
class WRTO:
    """WRTO ; Weighted least squares regression through the origin

    I could not find any decent Python package that does Weighted regression 
    through origin that also produces summary statistics, so I wrote my own.
    It is a fairly simple class that also keep package dependencies down.

    Parameters:    
        X : numpy.array or pandas.Series
            An array representing the independent observations of the regression.
        y : numpy.array or pandas.Series
            An array representing the dependent observations of the regression.
        w : numpy.array or pandas.Series
            An array representing the weights of the observations of the regression.
        
    Attributes:
        X : numpy.array or pandas.Series
            An array representing the independent observations of the regression.
        y : numpy.array or pandas.Series
            An array representing the dependent observations of the regression.
        w : numpy.array or pandas.Series
            An array representing the weights of the observations of the regression.
        coef : numpy.float64
            Slope parameter of the regression.
        WSSResidual : numpy.float64
            Weighted residual sum of squares of the regression.
        mse : numpy.float64
            Mean square error of the regression.
        se : numpy.float64
            Standard error of the regression slope paramter.
        sigma : numpy.float64
            Square root of the mean square error of the regression. 
        
    """
    def __init__(self,X,y,w):
        self.X = X
        self.y = y
        self.w = w
        self.coef_ = sum(w*X*y)/sum(w*X*X)
        self.WSSResidual = sum(w*((y-self.coef_*X)**2))
        self.mse = self.WSSResidual / (len(X)-1)
        self.se = np.sqrt(self.mse/sum(w*X*X))
        self.sigma = np.sqrt(self.mse)
        

class ChainLadder:
    def __init__(self, tri, weights=1, delta=1):
        self.triangle = tri
        weights = self.__checkWeights(weights)
        delta = [delta for item in range(len(self.triangle.data.columns)-1)]
        self.delta = delta
        self.weights = Triangle(weights)     
        self.models = self.fit()
        
        
    def __checkWeights(self, weights):
        return self.triangle.data*0 + weights
    
    def predict(self):
        ldf = [item.coef_ for item in self.models]
        square = self.triangle.data.copy()
        for row in range(1,len(square)):
            for col in range(row,0,-1):
                square.iloc[row,-col] = square.iloc[row,-col-1]*ldf[-col]
        return square
    
    def fit(self):
        models = []
        for i in range(0, len(self.triangle.data.columns)-1):
            #lm = LinearRegression(fit_intercept=False)
            data = self.triangle.data.iloc[:,i:i+2].dropna()
            w = self.weights.data.iloc[:,i].dropna().iloc[:-1]
            X = data.iloc[:,0].values
            y = data.iloc[:,1].values  
            sample_weight=w/X**self.delta[i]
            #lm.fit(X.reshape(-1, 1),y, sample_weight=w/X**self.delta[i])
            lm = WRTO(X,y,sample_weight)
            models.append(lm)   
        return models


class MackChainLadder:
    def __init__(self, tri,  weights=1,  alpha=1,  est_sigma="log-linear",  
                 tail=False,  tail_se=None,  tail_sigma=None,  mse_method = "Mack"):
        delta = 2-alpha
        cl = ChainLadder(tri, weights=weights,  delta=delta)
        alpha = [2 - item for item in cl.delta]
        
        self.triangle = tri
        self.weights = cl.weights
        self.fullTriangle = cl.predict()
        self.alpha = alpha
        self.f = np.array([item.coef_ for item in cl.models])
        self.fse = np.array([item.se for item in cl.models])
        self.sigma = np.array([item.sigma for item in cl.models])
        self.Fse = self.Fse()


    def process_risk(self):
        procrisk = DataFrame([0 for item in range(len(self.fullTriangle))],index=self.fullTriangle.index, columns=[self.fullTriangle.columns[0]])
        for i in range(1,len(self.fullTriangle.columns)):
            print(i)
            temp = DataFrame(np.sqrt((self.fullTriangle.iloc[:,i-1]*self.Fse.iloc[:,i-1])**2 + (self.f[i-1]*procrisk.iloc[:,i-1])**2)*self.triangle.data.iloc[:,i].isnull())
            temp.columns = [self.fullTriangle.columns[i]]
            procrisk = concat([procrisk, temp],axis=1)
        return procrisk

    def parameter_risk(self):
        paramrisk = DataFrame([0 for item in range(len(self.fullTriangle))],index=self.fullTriangle.index, columns=[self.fullTriangle.columns[0]])
        for i in range(1,len(self.fullTriangle.columns)):
            print(i)
            temp = DataFrame(np.sqrt((self.fullTriangle.iloc[:,i-1]*self.fse[i-1])**2 + (self.f[i-1]*paramrisk.iloc[:,i-1])**2)*self.triangle.data.iloc[:,i].isnull())
            temp.columns = [self.fullTriangle.columns[i]]
            paramrisk = concat([paramrisk, temp],axis=1)
        return paramrisk
    
    def Mack_SE(self):
        return  DataFrame(np.sqrt(np.matrix(self.process_risk()**2 )+np.matrix(self.parameter_risk()**2)), index=self.fullTriangle.index)
    
    def Fse(self):
        # This is sloppy, and I don't know that it works for all cases.  Need to
        # understand weights better.
        fulltriangleweightconst = self.weights.data.mode().T.mode().iloc[0,0]
        fulltriangleweight = self.fullTriangle*0 + fulltriangleweightconst
        Fse = DataFrame()
        for i in range(self.fullTriangle.shape[1]-2):
            Fse = concat([Fse, DataFrame(self.sigma[i]/np.sqrt(fulltriangleweight.iloc[:,i]*self.fullTriangle.iloc[:,i]**self.alpha[i]))],axis=1)
          
        Fse.set_index(self.fullTriangle.index, inplace = True)
        return Fse