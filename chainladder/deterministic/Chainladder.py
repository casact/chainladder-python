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
import itertools

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

    def __init__(self, triangle, weights=1, **kwargs):
        if isinstance(triangle, Triangle):
            self.triangle = copy.deepcopy(triangle).data_as_triangle()
        else:
            self.triangle = Triangle(data=triangle).data_as_triangle()
        if not isinstance(weights, Triangle):
            self.weights = Triangle(self.triangle.data * 0 + weights)
        else:
            self.weights = weights
        LDF_average = kwargs.get('LDF_average','volume')
        if type(LDF_average) is str:
            self.LDF_average = [LDF_average]*(self.triangle.ncol - 1)
        else:        
            self.LDF_average = np.array(LDF_average)[:self.triangle.ncol - 1]
        self.method = kwargs.get('method','DFM')
        self.apriori = kwargs.get('apriori',None)
        self.exposure = kwargs.get('exposure',None)
        self.trend = kwargs.get('trend',0)
        self.decay = kwargs.get('decay',1)
        self.n_iters = kwargs.get('n_iters',1)
        self.tail_factor = 1
        self.triangle_pred = kwargs.get('triangle_pred',None)
        self.set_LDF(self.LDF_average, inplace=True)
        if self.method == 'DFM': self.DFM(self.triangle_pred, inplace=True)
        if self.method == 'born_ferg': self.born_ferg(self.exposure, self.apriori, self.triangle_pred, inplace=True)
        if self.method == 'benktander': self.benktander(self.exposure, self.apriori, self.n_iters, self.triangle_pred, inplace=True)
        if self.method == 'cape_cod': self.cape_cod(self.exposure, self.trend, self.decay, self.triangle_pred, inplace=True)
        
    def __repr__(self):   
        return str(self.age_to_age())
             
    
    def set_LDF(self, LDF_average = 'volume', colname_sep='-', inplace=False):
        if inplace == True:
            tri_array = np.array(self.triangle.data)
            weights = np.array(self.weights.data)
            x = np.nan_to_num(tri_array[:,:-1]*(tri_array[:,1:]*0+1))
            w = np.nan_to_num(weights[:,:-1]*(tri_array[:,1:]*0+1))
            y = np.nan_to_num(tri_array[:,1:])
            if type(LDF_average) is str:
                LDF_average = [LDF_average]*(self.triangle.ncol - 1)
            self.LDF_average = LDF_average
            average = np.array(self.LDF_average)
            #Volume Weighted average and Straight Average, and regression through origin
            val = np.repeat((np.array([[{'regression':0, 'volume':1,'simple':2}.get(item.lower(),2) for item in average]])),len(self.triangle.data.index),axis=0)
            val = np.nan_to_num(val*(tri_array[:,1:]*0+1))
            w = np.nan_to_num(w/tri_array[:,:-1]**(val))
            LDF = np.sum(w*x*y,axis=0)/np.sum(w*x*x,axis=0)
            
            #Harmonic Mean
            harmonic = np.sum(np.nan_to_num(tri_array[:,1:]*0+1),axis=0)/np.sum(np.reciprocal(w*x*y, where=w*x*y!=0),axis=0)
            #Geometric Mean
            geometric = np.prod(w*x*y+np.logical_not(np.nan_to_num(tri_array[:,1:]*0+1)),axis=0)**(1/np.sum(np.nan_to_num(tri_array[:,1:]*0+1),axis=0))
            
            LDF = LDF*(average=='volume')+LDF*(average=='simple')+LDF*(average=='regression')+geometric*(average=='geometric')+harmonic*(average=='harmonic')            
            LDF = np.append(LDF,[self.tail_factor]) # Need to modify for tail
            columns = [str(item) + colname_sep + str(self.triangle.data.columns.values[num + 1]) 
                    for num, item in enumerate(self.triangle.data.columns.values[:-1])] + [str(self.triangle.data.columns.values[-1]) + colname_sep + 'Ult']
            self.LDF = Series(LDF, index=columns)
            CDF = np.cumprod(LDF[::-1])[::-1]
            self.CDF = Series(CDF, index=columns)
            return self
        if inplace == False:
            new_instance = copy.deepcopy(self)
            return new_instance.set_LDF(LDF_average=LDF_average, colname_sep = colname_sep, inplace=True)
        
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
        if self.triangle.dataform == 'tabular':
            x = self.triangle.data_as_triangle().data
        else:
            x = self.triangle.data
        incr = DataFrame(
                np.array(x.iloc[:,1:])/
                np.array(x.iloc[:,:-1]),
                index=x.index, columns = 
                [str(item) + colname_sep + str(x.columns.values[num + 1]) 
                for num, item in enumerate(x.columns.values[:-1])])
        incr[str(x.columns.values[-1]) + colname_sep + 'Ult'] = np.nan
        incr.iloc[0,-1]=1
        #print(len(incr.columns))
        #print(len(self.LDF_average))
        incr.loc['Average'] = list(self.LDF_average)+['tail']
        incr.loc['LDF'] = self.LDF
        incr.loc['CDF'] = self.CDF
        return incr.round(4)
    
    
    def grid_search(self, param_grid):
        for my_dict in param_grid:
            my_list = [param_grid[my_dict][key] for key in param_grid[my_dict]]
            key_list = list(param_grid[my_dict].keys())
            value_list = []
            for element in itertools.product(*my_list):
                value_list.append(element)
            for item in value_list:
               grid_dict = dict(zip(key_list, item))
               #print('Method: ' + my_dict + '\n' + 'Paremeters:\n'+str(grid_dict))
               print(Chainladder(self.triangle, method=my_dict, **grid_dict).ultimates)
           
    def DFM(self, triangle = None, inplace = False):
        """ Method to 'square' the triangle based on the WRTO 'models' list.
        
        Returns:
            pandas.DataFrame representing the raw triangle data as well as future
            lags populated with the expectation from the chainladder fit.
        """
        
        if inplace==True:
            if triangle is None:
                triangle =self.triangle
            latest = np.array(triangle.get_latest_diagonal())
            self.ultimates = Series((latest[:,1]*np.array(self.CDF)[::-1]), index = triangle.data.index)
            return self
        if inplace==False:
            new_instance = copy.deepcopy(self)
            return new_instance.DFM(triangle=triangle, inplace=True)
            
    
    def born_ferg(self, exposure, apriori, triangle = None, inplace=False):
        if inplace == True:
            if triangle is None:
                triangle = self.triangle
            if type(apriori) in [float, int]:
                my_apriori = np.array([apriori]*len(exposure))
            my_apriori = np.array(exposure) * apriori
            latest = np.array(triangle.get_latest_diagonal())
            CDF = np.array(self.CDF)[latest[:,0].astype(int)-1]
            unreported_pct = 1 - 1 / CDF
            self.ultimates = Series(latest[:,1] + unreported_pct * my_apriori,index=triangle.data.index)
            self.method = 'born_ferg'
            self.apriori = apriori
            self.exposure = exposure
            return self
        if inplace == False:
            new_instance = copy.deepcopy(self)
            return new_instance.born_ferg(exposure=exposure, apriori = apriori, triangle=triangle, inplace=True)
    
    def benktander(self, exposure, apriori, n_iters = 1, triangle = None, inplace=False):
        if inplace == True:
            if triangle is None:
                triangle = self.triangle
            temp_ult = np.array(exposure) * apriori
            for i in range(n_iters):
                temp_ult = self.born_ferg(exposure=temp_ult, apriori=1, triangle=triangle).ultimates
            self.ultimates = temp_ult
            self.method = 'benktander'
            self.apriori = apriori
            self.exposure = exposure
            self.n_iters = n_iters
            return self
        if inplace == False:
            new_instance = copy.deepcopy(self)
            return new_instance.benktander(exposure=exposure, apriori = apriori, n_iters = n_iters, triangle = triangle, inplace=True)
    
    def cape_cod(self, exposure, trend = 0, decay = 1, triangle= None, inplace=False):
        if inplace == True:
            if triangle is None:
                triangle = self.triangle
            latest = triangle.get_latest_diagonal()
            latest = np.array(latest[exposure>0])
            CDF = np.array(self.CDF)[latest[:,0].astype(int)-1]
            reported_exposure = np.array(exposure[exposure>0])/ CDF
            trend_array = np.array([(1+trend)**(sum(exposure>0) - (i+1)) for i in range(sum(exposure>0))])
            decay_matrix = np.array([[decay**abs(i-j) for i in range(sum(exposure>0) )] for j in range(sum(exposure>0))])
            weighted_exposure = reported_exposure * decay_matrix
            trended_ultimate = np.repeat(np.array([(latest[:,1] * trend_array) /(reported_exposure)]),sum(exposure>0),axis=0)
            apriori = np.sum(weighted_exposure*trended_ultimate,axis=1)/np.sum(weighted_exposure,axis=1)
            detrended_ultimate = apriori/trend_array
            IBNR = detrended_ultimate * (1-1/CDF) * np.array(exposure[exposure>0])
            self.ultimates = Series(latest[:,1] + IBNR, index=exposure[exposure>0].index)
            self.method = 'cape_cod'
            self.exposure = exposure
            self.trend=trend
            self.decay = decay
            return self
        if inplace == False:
            new_instance = copy.deepcopy(self)
            return new_instance.cape_cod(exposure=exposure, trend = trend, decay = decay, triangle=triangle, inplace=True)
        
    def berq_serm():
        return
    
    def fish_lang():
        return
    
    
    def exclude_link_ratios(self, otype, inplace=False):
        if inplace == True:
            if type(otype) is list:
                if len(otype)>0:
                    if type(otype[0]) is not tuple:
                        otype = [(item[0],item[1]) for item in otype]
                for item in otype:
                    self.weights.data.iloc[item] = 0
            else:    
                ata = self.age_to_age().iloc[:-3]
                ata.index = self.weights.data.index
                ata.columns = self.weights.data.columns
                if otype == 'min':
                    val = np.invert(ata == ata.min())
                if otype == 'max':
                    val = np.invert(ata == ata.max())
                self.weights.data = self.weights.data * val
            self.set_LDF(self.LDF_average, inplace=True)
            if self.method == 'DFM': self.DFM(self.triangle_pred, inplace=True)
            if self.method == 'born_ferg': self.born_ferg(self.exposure, self.apriori, self.triangle_pred, inplace=True)
            if self.method == 'benktander': self.benktander(self.exposure, self.apriori, self.n_iters, self.triangle_pred, inplace=True)
            if self.method == 'cape_cod': self.cape_cod(self.exposure, self.trend, self.decay, self.triangle_pred, inplace=True)
            return self
        if inplace == False:
            new_instance = copy.deepcopy(self)
            return new_instance.exclude_link_ratios(otype=otype, inplace=True)
            
    def exponential_tail(self, colname_sep='-'):
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
        LDF = self.LDF
        if len(LDF[LDF>1]) < 2:
            warn("Not enough factors larger than 1.0 to fit an exponential regression.")
            tail_factor = 1
        elif (LDF[-3] * LDF[-2] > 1.0001):
            y = Series(np.array(LDF)[:-1])
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
        self.tail_factor = tail_factor
        self.set_LDF(self.LDF_average, inplace=True)
        return self
        

        
        