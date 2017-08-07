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
from pandas import DataFrame, concat, Series
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence
from warnings import warn
from chainladder.Triangle import Triangle
import copy

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
        tri : `Triangle <Triangle.html>`_
            A triangle object. Refer to :class:`Classes.Triangle`
        weights : int
            A value representing an input into the weights of the WRTO class.
        delta : int
            A value representing an input into the weights of the WRTO class.
        tail : bool
            A value representing whether the user would like an exponential tail
            factor to be applied to the data.

    Attributes:
        tri : `Triangle <Triangle.html>`_
            A triangle object on which the Chainladder model will be built.
        weights : pandas.DataFrame
            A value representing an input into the weights of the WRTO class.
        delta : list
            A set of values representing an input into the weights of the WRTO 
            class.
        models : list
            A list of WTRO objects for of length (col-1)
        tail : bool
            A value representing whether the user would like an exponential tail
            factor to be applied to the data.
        LDF : pandas.Series
            A series containing the LDFs of the chainladder model.  If tail=True
            and tail fitting succeeds, an additional 'Age-Ult' factor is appended
            the the end representing the tail CDF.
        CDF : pandas.Series
            A series representing the cumulative development factors of the
            chainladder model.
        full_triangle : Pandas.DataFrame
            A table representing the raw triangle data as well as future
            lags populated with the expectation from the chainladder fit.

    """

    def __init__(self, tri, weights=1, delta=1, tail=False):
        if type(tri) is Triangle:
            self.triangle = copy.deepcopy(tri)
        elif tri.shape[1]>= tri.shape[0]:
            self.triangle = Triangle(tri)
        self.delta = [delta for item in range(self.triangle.ncol)]
        self.weights = Triangle(self.triangle.data * 0 + weights)
        self.models = self.fit()
        self.tail = tail
        self.LDF = self.get_LDF().append(self.get_tail_factor())
        self.CDF = self.LDF[::-1].cumprod()[::-1]
        self.full_triangle = self.predict()
        
    def __repr__(self):   
        return str(self.age_to_age())
    
    def predict(self):
        """ Method to 'square' the triangle based on the WRTO 'models' list.
        
        Returns:
            pandas.DataFrame representing the raw triangle data as well as future
            lags populated with the expectation from the chainladder fit.
        """
        ldf = [item.coefficient for item in self.models]
        square = self.triangle.data.copy()
        for row in range(1, len(square)):
            for col in range(row, 0, -1):
                square.iloc[row, -col] = square.iloc[row, -col - 1] * ldf[-col]
        square['Ult']=square.iloc[:,-1]*self.CDF[-1]
        return square

    def fit(self):
        """ Method to call the weighted regression trhough the origin fitting.
        
        Returns:
            list of statsmodel (Chainladder) models with a subset of properties 
            that can be accessed later.  See WRTO class implementation.
        """
        models = []
        for i in range(0, self.triangle.ncol - 1):
            data = self.triangle.data.iloc[:, i:i + 2].dropna()
            w = self.weights.data.iloc[:, i].dropna().iloc[:-1]
            X = data.iloc[:, 0]
            y = data.iloc[:, 1]
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
        for i in range(1, self.triangle.ncol - 1):
            incr = concat([incr, self.triangle.data.iloc[:, i + 1] /
                           self.triangle.data.iloc[:, i]], axis=1)
        incr.columns = [str(item) + colname_sep +
                        str(self.triangle.data.columns.values[num + 1]) for num,
                        item in enumerate(self.triangle.data.columns.values[:-1])]
        incr = incr.iloc[:-1]
        incr[str(self.triangle.data.columns.values[-1]) + colname_sep + 'Ult'] = np.nan
        ldf_s = Chainladder(self.triangle.data, delta=2, tail=self.tail).LDF
        ldf_v = Chainladder(self.triangle.data, delta=1, tail=self.tail).LDF
        if len(incr.index.names)>1:
            incr.loc[tuple("simple" if i == 0 else "" for i in range(0,len(incr.index.names))),:] = list(ldf_s)
            incr.loc[tuple("vol-wtd" if i == 0 else "" for i in range(0,len(incr.index.names))),:] = list(ldf_v)
            incr.loc[tuple("Selected" if i == 0 else "" for i in range(0,len(incr.index.names))),:] = list(ldf_v)
        else:
            incr.loc['simple'] = list(ldf_s)
            incr.loc["vol-wtd"] = list(ldf_v)
            incr.loc["Selected"] = list(ldf_v)
        ldf = self.LDF
        incr.iloc[0,-1]=1.0
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
        LDF = Series([ldf.coefficient for ldf in self.models], index=[str(item) + colname_sep +
                        str(self.triangle.data.columns.values[num + 1]) for num,
                        item in enumerate(self.triangle.data.columns.values[:-1])])
        if len(LDF) >= self.triangle.ncol-1:
            return Series(LDF[:self.triangle.ncol-1])
        else:
            return LDF
            
         
        

    def get_tail_factor(self, colname_sep='-'):
        """Estimate tail factor, idea from Thomas Mack:
        Returns a tail factor based off of an exponential fit to the LDFs.  This will
        fail if the product of 2nd and 3rd to last LDF < 1.0001.  This also fails if
        the estimated tail is larger than 2.0.  In other areas, exponential fit is
        rejected if the slope parameter p-value >0.5.  This is currently representative
        of the R implementation of this package, but may be enhanced in the future to be
        p-value based.

        Parameters:    
            colname_sep : str
                text to join the names of two adjacent columns representing the
                age-to-age factor column name.

        Returns:
            Pandas.Series of the tail factor.        
        """
        LDF = np.array(self.get_LDF()[:self.triangle.ncol-1])
        if self.tail==False:
            tail_factor=1
        elif len(LDF[LDF>1]) < 2:
            warn("Not enough factors larger than 1.0 to fit an exponential regression.")
            tail_factor = 1
        elif (LDF[-3] * LDF[-2] > 1.0001):
            y = Series(LDF)
            x = sm.add_constant((y.index+1)[y>1])
            y = LDF[LDF>1]
            n, = np.where(LDF==y[-1])[0]
            tail_model = sm.OLS(np.log(y-1),x).fit()
            tail_factor = np.product(np.exp(tail_model.params[0] + np.array([i+2 for i in range(n,n+100)]) * tail_model.params[1]).astype(float) + 1)
            if tail_factor > 2:
                warn("The estimate tail factor was bigger than 2 and has been reset to 1.")
                tail_factor = 1
            if tail_model.f_pvalue > 0.05:
                warn("The p-value of the exponential tail fit is insignificant and tail has been set to 1.")
                tail_factor = 1
        else:
            warn("LDF[-2] * LDF[-1] is not greater than 1.0001 and tail has been set to 1.")
            tail_factor = 1
        return Series(tail_factor, index = [str(self.triangle.data.columns[-1]) + colname_sep + 'Ult'])

    def get_residuals(self):
        """Generates a table of chainladder residuals along with other statistics.
        These get used in the MackChainLadder.plot() method for residual plots.
         
        Returns:
            Pandas.DataFrame of the residual table        
        """
        
        Resid = DataFrame()
        for i in range(len(self.models)):
            resid = DataFrame()
            resid['x'] = self.models[i].x
            resid['dev_lag'] = int(self.models[i].x.name)
            resid['dev_lag'] = resid['dev_lag']
            resid['residuals'] = self.models[i].residual
            resid['standard_residuals'] = self.models[i].std_resid
            resid['fitted_value'] = self.models[i].fittedvalues
            Resid = Resid.append(resid)
        Resid = Resid.reset_index().merge(self.triangle.lag_to_date().reset_index(), how='inner', on=[self.triangle.origin_dict[key] for key in self.triangle.origin_dict]+['dev_lag'])
        return Resid.set_index([self.triangle.origin_dict[key] for key in self.triangle.origin_dict]).drop(['x'], axis=1).dropna()
    
class WRTO():
    """Weighted least squares regression through the origin

    Collecting the relevant variables from statsmodel OLS/WLS. Note in
    release 0.1.0 of chainladder, there is a deprecation warning with statsmodel
    that will persist until statsmodel is upgraded.

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
        residual : numpy.array
            The difference between actual y value and the fitted/expected y value
        standard_error : numpy.float64
            Standard error of the regression slope paramter.
        sigma : numpy.float64
            Square root of the mean square error of the regression. 
        std_resid : numpy.array
            Represents internally studentized residuals which generally vary between
            [-2,2].  Used in residual scatterplots and help determine the appropriateness
            of the model on the data.

    """

    def __init__(self, x, y, w):
        self.x = x
        self.y = y
        self.w = w
        WLS = sm.WLS(y,x, w)
        OLS = sm.OLS(WLS.wendog,WLS.wexog).fit()
        self.coefficient = OLS.params[0]
        self.WSSResidual = OLS.ssr
        self.fittedvalues = OLS.predict(x)
        self.residual = OLS.resid
        if len(x) == 1:
            self.mean_square_error = np.nan 
            self.standard_error = np.nan
            self.sigma = np.nan
            self.std_resid = np.nan
        else:
            self.mean_square_error = OLS.mse_resid 
            self.standard_error = OLS.params[0]/OLS.tvalues[0]
            self.sigma = np.sqrt(self.mean_square_error)
            self.std_resid = OLSInfluence(OLS).resid_studentized_internal
        
        
        
        