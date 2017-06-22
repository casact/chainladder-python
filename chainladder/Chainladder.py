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
from pandas import DataFrame, concat, pivot_table, Series
from warnings import warn
from chainladder.Triangle import Triangle

class Chainladder():
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

    def __init__(self, tri, weights=1, delta=1, tail=False):
        if type(tri) is Triangle:
            self.triangle = tri 
        else:
            self.triangle = Triangle(tri)
        self.delta = [delta for item in range(len(self.triangle.data.columns) )]
        self.weights = Triangle(self.triangle.data * 0 + weights)
        self.models = self.fit()
        self.tail = tail
        self.LDF = self.get_LDF().append(self.get_tail_factor())
        self.CDF = self.LDF[::-1].cumprod()[::-1]
        self.full_triangle = self.predict()

    def predict(self):
        """ Method to 'square' the triangle based on the 'models' list.
        """
        ldf = [item.coefficient for item in self.models]
        square = self.triangle.data.copy()
        for row in range(1, len(square)):
            for col in range(row, 0, -1):
                square.iloc[row, -col] = square.iloc[row, -col - 1] * ldf[-col]
        square['Ult']=square.iloc[:,-1]*self.CDF[-1]
        return square

    def fit(self):
        """ Method to call the weighted regression trhough the origin fitting .
        """
        models = []
        for i in range(0, len(self.triangle.data.columns) - 1):
            data = self.triangle.data.iloc[:, i:i + 2].dropna()
            w = self.weights.data.iloc[:, i].dropna().iloc[:-1]
            X = data.iloc[:, 0].values
            y = data.iloc[:, 1].values
            sample_weight = np.array(w).astype(
                float) / np.array(X).astype(float)**self.delta[i]
            lm = WRTO(X, y, sample_weight)
            models.append(lm)
        return models

    def age_to_age(self, colname_sep='-'):
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
        incr = DataFrame(self.triangle.data.iloc[:, 1] /
                         self.triangle.data.iloc[:, 0])
        for i in range(1, len(self.triangle.data.columns) - 1):
            incr = concat([incr, self.triangle.data.iloc[:, i + 1] /
                           self.triangle.data.iloc[:, i]], axis=1)
        incr.columns = [item + colname_sep +
                        self.triangle.data.columns.values[num + 1] for num,
                        item in enumerate(self.triangle.data.columns.values[:-1])]
        incr = incr.iloc[:-1]
        incr[str(self.triangle.data.columns.values[-1]) + colname_sep + 'Ult'] = np.nan
        ldf = Chainladder(self.triangle.data, delta=2, tail=self.tail).LDF
        incr.loc['simple'] = ldf
        ldf = Chainladder(self.triangle.data, delta=1, tail=self.tail).LDF
        incr.loc['vol-wtd'] = ldf
        incr.loc['Selected'] = ldf
        ldf = self.LDF
        return incr
    
    def get_LDF(self, colname_sep='-'):
        """ Method to obtain the loss development factors (LDFs) from the
        chainladder model.
        
        Parameters:    
            colname_sep : str
                text to join the names of two adjacent columns representing the
                age-to-age factor column name.

        Returns:
            Pandas.Series of the LDFs.
        
        """
        LDF = Series([ldf.coefficient for ldf in self.models], index=[item + colname_sep +
                        self.triangle.data.columns.values[num + 1] for num,
                        item in enumerate(self.triangle.data.columns.values[:-1])])
        if len(LDF) >= len(self.triangle.data.columns)-1:
            return LDF[:len(self.triangle.data.columns)-1]
        else:
            return LDF
            
         
        

    def get_tail_factor(self, colname_sep='-'):
        """Estimate tail factor, idea from Thomas Mack:
        THE STANDARD ERROR OF CHAIN LADDER RESERVE ESTIMATES:
        RECURSIVE CALCULATION AND INCLUSION OF A TAIL FACTOR

        Parameters:    
            colname_sep : str
                text to join the names of two adjacent columns representing the
                age-to-age factor column name.

        Returns:
            Pandas.Series of the tail factor.        
        """
        LDF = self.get_LDF()[:len(self.triangle.data.columns)-1]
        n = len(LDF)
        if (LDF[-1] * LDF[-2] > 1.0001) and self.tail == True:
            LDF_positive = np.array([item for item in LDF if item > 1])
            n = np.max(np.where(LDF >= 1)[0])            
            tail_model = np.polyfit(range(1,len(LDF_positive)+1),np.log(LDF_positive-1),1)
            tail_factor = np.product(np.exp(tail_model[1] + np.array([i+2 for i in range(n,n+100)]) * tail_model[0]).astype(float) + 1)
            if tail_factor > 2:
                warn("The estimate tail factor was bigger than 2 and has been reset to 1.")
                tail_factor = 1
        else:
            tail_factor = 1
        return Series(tail_factor, index = [self.triangle.data.columns[-1] + colname_sep + 'Ult'])
    
class WRTO():
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
        coefficient : numpy.float64
            Slope parameter of the regression.
        WSSResidual : numpy.float64
            Weighted residual sum of squares of the regression.
        mean_square_error : numpy.float64
            Mean square error of the regression.
        standard_error : numpy.float64
            Standard error of the regression slope paramter.
        sigma : numpy.float64
            Square root of the mean square error of the regression. 

    """

    def __init__(self, X, y, w):
        self.X = X
        self.y = y
        self.w = w
        self.coefficient = sum(w * X * y) / sum(w * X * X)
        self.WSSResidual = sum(w * ((y - self.coefficient * X)**2))
        if len(X) == 1:
            self.mean_square_error = np.nan 
        else:
            self.mean_square_error = self.WSSResidual / (len(X) - 1)
        self.standard_error = np.sqrt(self.mean_square_error / sum(w * X * X))
        self.sigma = np.sqrt(self.mean_square_error)
