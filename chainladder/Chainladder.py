# -*- coding: utf-8 -*-
"""
The Chainladder Module allows for the computation of basic chainladder methods 
on an object of the Triangle class.  The module contains two classes, 
ChainLadder and WRTO.  Chainladder is the main class while WRTO (Weighted 
regression through the origin) is a unique case of a linear model with 0 
intercept and a single slope parameter which is the form of a chainladder LDF
approach.
"""

import numpy as np
from pandas import DataFrame, concat, Series, pivot_table
from scipy import stats
from chainladder.Triangle import Triangle


class ChainLadder:
    """ ChainLadder class specifies the chainladder model.
    
    The classical chain-ladder is a deterministic algorithm to forecast claims 
    based on historical data. It assumes that the proportional developments of 
    claims from one development period to the next are the same for all origin 
    years.
    
    Mack uses alpha between 0 and 2 to distinguish
    alpha = 0 straight averages
    alpha = 1 historical chain ladder age-to-age factors
    alpha = 2 ordinary regression with intercept 0

    However, in Zehnwirth & Barnett they use the notation of delta, whereby 
    delta = 2 - alpha the delta is than used in a linear modelling context.
    
    `Need to properly cite... <https://github.com/mages/ChainLadder>`_
    
    Parameters:    
        tri : Triangle
            A triangle object. Refer to :class:`Classes.Triangle`
        weights : int
            A value representing an input into the weights of the WRTO class.
        delta : int
            A value representing an input into the weights of the WRTO class.
        
    Attributes:
        tri : Triangle
            A triangle object on which the Chainladder model will be built.
        weights : pandas.DataFrame
            A value representing an input into the weights of the WRTO class.
        delta : list
            A set of values representing an input into the weights of the WRTO 
            class.
        models : list
            A list of WTRO objects for of length (col-1)
    
    """
    def __init__(self, tri, weights=1, delta=1):
        self.triangle = tri
        weights = self.__checkWeights(weights)
        delta = [delta for item in range(len(self.triangle.data.columns)-1)]
        self.delta = delta
        self.weights = Triangle(weights)     
        self.models = self.fit()
        
        
    def __checkWeights(self, weights):
        """ Hidden method used to convert weights from a scalar into a matrix 
        of the same shape as the triangle
        """
    
        return self.triangle.data*0 + weights
    
    def predict(self):
        """ Method to 'square' the triangle based on the 'models' list.
        """
        ldf = [item.coef_ for item in self.models]
        square = self.triangle.data.copy()
        for row in range(1,len(square)):
            for col in range(row,0,-1):
                square.iloc[row,-col] = square.iloc[row,-col-1]*ldf[-col]
        return square
    
    def fit(self):
        """ Method to call the weighted regression trhough the origin fitting .
        """
        models = []
        for i in range(0, len(self.triangle.data.columns)-1):
            #lm = LinearRegression(fit_intercept=False)
            data = self.triangle.data.iloc[:,i:i+2].dropna()
            w = self.weights.data.iloc[:,i].dropna().iloc[:-1]
            X = data.iloc[:,0].values
            y = data.iloc[:,1].values  
            sample_weight=np.array(w).astype(float)/np.array(X).astype(float)**self.delta[i]
            lm = WRTO(X,y,sample_weight)
            models.append(lm)   
        return models
    
    def ata(self, colname_sep = '-'):
        """ Method to display an age-to-age triangle with a display of simple 
        average chainladder development factors and volume weighted average 
        development factors.
        
        Parameters:    
            colname_sep : str
                text to join the names of two adjacent columns representing the
                age-to-age factor column name.
        
        Returns:
            Pandas.DataFrame of the age-to-age triangle.
        """
        incr = DataFrame(self.triangle.data.iloc[:,1]/
                         self.triangle.data.iloc[:,0])
        for i in range(1, len(self.triangle.data.columns)-1):
            incr = concat([incr,self.triangle.data.iloc[:,i+1]/
                           self.triangle.data.iloc[:,i]],axis=1)
        incr.columns = [item + colname_sep + 
                        self.triangle.data.columns.values[num+1] for num, 
                        item in enumerate(self.triangle.data.columns.values[:-1])]
        incr = incr.iloc[:-1]
        ldf = [item.coef_ for item in ChainLadder(self.triangle, delta=2).models]
        incr.loc['smpl']=ldf
        ldf = [item.coef_ for item in ChainLadder(self.triangle, delta=1).models]
        incr.loc['vwtd']=ldf
        return incr.round(3)

class WRTO:
    """Weighted least squares regression through the origin

    I could not find any decent Python package that does Weighted regression 
    through origin that also produces summary statistics, so I wrote my own.
    It is a fairly simple class that also keep package dependencies down.

    Parameters:    
        X : numpy.array or pandas.Series
            An array representing the independent observations of the 
            regression.
        y : numpy.array or pandas.Series
            An array representing the dependent observations of the regression.
        w : numpy.array or pandas.Series
            An array representing the weights of the observations of the 
            regression.
        
    Attributes:
        X : numpy.array or pandas.Series
            An array representing the independent observations of the 
            regression.
        y : numpy.array or pandas.Series
            An array representing the dependent observations of the regression.
        w : numpy.array or pandas.Series
            An array representing the weights of the observations of the 
            regression.
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
        if len(X) == 1:
            self.mse = np.nan
        else:
            self.mse = self.WSSResidual / (len(X)-1)
        self.se = np.sqrt(self.mse/sum(w*X*X))
        self.sigma = np.sqrt(self.mse)   