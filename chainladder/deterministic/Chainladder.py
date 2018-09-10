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
from pandas import DataFrame, concat, Series, to_datetime
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence
from warnings import warn
from chainladder.Triangle import Triangle
import copy
import itertools
import holoviews as hv

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
            self.triangle = copy.deepcopy(triangle)
        else:
            self.triangle = Triangle(data=triangle.copy())
        self.triangle.ncol = len(self.triangle.data.columns)
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
        self.set_LDF(self.LDF_average, tail_factor=self.tail_factor, inplace=True)
        if self.method == 'DFM': self.DFM(self.triangle_pred, inplace=True)
        if self.method == 'born_ferg': self.born_ferg(self.exposure, self.apriori, self.triangle_pred, inplace=True)
        if self.method == 'benktander': self.benktander(self.exposure, self.apriori, self.n_iters, self.triangle_pred, inplace=True)
        if self.method == 'cape_cod': self.cape_cod(self.exposure, self.trend, self.decay, self.triangle_pred, inplace=True)

    def __repr__(self):
        return str(self.age_to_age())

    def set_LDF(self, LDF_average = None, colname_sep='-', tail_factor=1.0, inplace=False):
        """Method to convert triangle form to tabular form.

        Arguments:
            LDF_average: str or list
                Allows for the selection of 'volume', 'simple' 'regression', 'geometric', and 'harmonic averages.
                Note that 'regression' is an OLS regression through the origin.  Specified as string will apply the
                same average to all development periods.  Specified as a list, each LDF can take on a different average.
                As a list, the length must be equal to the number of columns in the triangle - 1.



        Returns:
            Updated instance `data` parameter if inplace is set to True otherwise it returns a pandas.DataFrame
        """
        if inplace == True:
            if LDF_average is None:
                LDF_average = self.LDF_average
            tri_array = np.array(self.triangle.data)
            # set tri_array 0s to 1s and set weights to 0
            weights = np.array(self.weights.data)
            x = np.nan_to_num(tri_array[:,:-1]*(tri_array[:,1:]*0+1))
            #w = np.nan_to_num(weights[:,:-1]*(tri_array[:,1:]*0+1))
            w = np.nan_to_num(weights[:,:-1]*(np.minimum(np.nan_to_num(tri_array[:,1:]),1)*(tri_array[:,1:]*0+1)))
            #(np.maximum(np.nan_to_num(tri_array[:,:-1]),1)*(tri_array[:,:-1]*0+1))
            y = np.nan_to_num(tri_array[:,1:])
            if type(LDF_average) is str:
                LDF_average = [LDF_average]*(self.triangle.ncol - 1)
            self.LDF_average = LDF_average
            average = np.array(LDF_average)
            #Volume Weighted average and Straight Average, and regression through origin
            val = np.repeat((np.array([[{'regression':0, 'volume':1,'simple':2}.get(item.lower(),2) for item in average]])),len(self.triangle.data.index),axis=0)
            val = np.nan_to_num(val*(tri_array[:,1:]*0+1))
            #w = np.nan_to_num(w/tri_array[:,:-1]**(val))
            w = np.nan_to_num(w/(np.maximum(np.nan_to_num(tri_array[:,:-1]),1)*(tri_array[:,:-1]*0+1))**(val))
            LDF = np.sum(w*x*y,axis=0)/np.sum(w*x*x,axis=0)
            #Harmonic Mean
            w = np.array(self.weights.data.iloc[:,:-1])
            w = np.array(self.weights.data.iloc[:,:-1])
            w_geom = np.logical_not(np.logical_not(w))
            w_harm = np.nan_to_num(w)
            x[x == 0] = np.inf
            ata = y/x*w_harm
            harmonic = (np.sum(w_harm,axis=0)-1)/np.sum(1/np.where(ata==0,1,ata)*(ata!=0),axis=0)

            #Geometric Mean
            w_geom = np.logical_not(np.logical_not(w))
            geometric = ata
            geometric[geometric == 0] = 1


            geometric = np.prod(geometric,axis=0)**(1/(np.sum(w_geom,axis=0)-1))
            # Array of product of Series is due to bypass an invalid numpy warning - weird code...but it works
            LDF = LDF*(average=='volume')+LDF*(average=='simple')+LDF*(average=='regression')+np.array(Series(average=='geometric')*Series(geometric))+np.array(Series(harmonic)*Series(average=='harmonic'))
            self.tail_factor = tail_factor
            LDF = np.append(LDF,[self.tail_factor])
            columns = [str(item) + colname_sep + str(self.triangle.data.columns.values[num + 1])
                    for num, item in enumerate(self.triangle.data.columns.values[:-1])] + [str(self.triangle.data.columns.values[-1]) + colname_sep + 'Ult']
            self.LDF = Series(LDF, index=columns)
            CDF = np.cumprod(LDF[::-1])[::-1]
            self.CDF = Series(CDF, index=columns)
            return self
        if inplace == False:
            new_instance = copy.deepcopy(self)
            return new_instance.set_LDF(LDF_average=LDF_average, colname_sep = colname_sep, tail_factor=tail_factor, inplace=True)

    def select_average(self, LDF_average = None, num_periods=None, tail_factor=1.0, inplace=False):
        """ Method to select number of years and the average used in LDF pick of the chainladder model.

         Arguments:
            LDF_average: str or list
                Allows for the selection of 'volume', 'simple' 'regression', 'geometric', and 'harmonic averages.
                Note that 'regression' is an OLS regression through the origin.  Specified as string will apply the
                same average to all development periods.  Specified as a list, each LDF can take on a different average.
                As a list, the length must be equal to the number of columns in the triangle - 1.
            num_periods: int or list
                Specify the number of origin periods to use in the LDF averaging.  Specified as a list, each LDF can
                take on a number of periods, and the list length must be equal to the number of columns in the triangle - 1.
            inplace: bool
                Set to true will mutate the Chainladder object, set to false will create a copy.

         Returns:
            Chainladder object with LDF parameters selected.

        """
        if inplace == True:
            temp = np.array(self.weights.data)
            total_index = temp.shape[0]
            if num_periods is None:
                num_periods = np.repeat(total_index,temp.shape[1])
            if type(num_periods) == int:
                num_periods = np.repeat(min(num_periods,total_index),temp.shape[1])
            for i in range(temp.shape[1] - 1):
                try:
                    first_index = max(np.where(np.isnan(temp[:,i])==True)[0][0] - num_periods[i] - 1 ,0)
                except:
                    first_index = max(total_index- num_periods[i] - 1 ,0)
                temp[:,i] = np.append(np.repeat(0,first_index),np.repeat(1,total_index-first_index))*(temp[:,i]*0+1)
            self.weights = Triangle(DataFrame(temp, index=self.weights.data.index, columns = self.weights.data.columns))
            self.set_LDF(LDF_average, tail_factor=tail_factor,inplace=True)
            if self.method == 'DFM': self.DFM(self.triangle_pred, inplace=True)
            if self.method == 'born_ferg': self.born_ferg(self.exposure, self.apriori, self.triangle_pred, inplace=True)
            if self.method == 'benktander': self.benktander(self.exposure, self.apriori, self.n_iters, self.triangle_pred, inplace=True)
            if self.method == 'cape_cod': self.cape_cod(self.exposure, self.trend, self.decay, self.triangle_pred, inplace=True)
            return self
        if inplace == False:
            new_instance = copy.deepcopy(self)
            return new_instance.select_average(LDF_average=LDF_average, num_periods=num_periods, tail_factor=tail_factor, inplace=True)

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

        x = self.triangle.data
        numer = np.array(x.iloc[:,1:])
        denom = np.array(x.iloc[:,:-1])
        denom[denom == 0] = np.inf
        incr = DataFrame(
                numer/denom,
                index=x.index, columns =
                [str(item) + colname_sep + str(x.columns.values[num + 1])
                for num, item in enumerate(x.columns.values[:-1])])
        incr[str(x.columns.values[-1]) + colname_sep + 'Ult'] = np.nan
        incr.iloc[0,-1]=1

        incr.loc['Average'] = list(self.LDF_average)+['tail']
        incr.loc['LDF'] = self.LDF
        incr.loc['CDF'] = self.CDF
        return incr.round(4)

    def DFM(self, triangle = None, inplace = False):
        """ Method to develop ultimates using the Development Factor Method (DFM)

        Parameters:
            triangle: Triangle or None
                Set as None will apply the DFM to the triangle embedded in the Chainladder object.
                This can be overriden to apply the DFM method to a new triangle.

            inplace: bool
                Set to true will mutate the Chainladder object, set to false will create a copy.

         Returns:
            Chainladder object fit with the DFM method.
        """

        if inplace == True:
            if triangle is None:
                triangle = self.triangle
            exposure = np.array(triangle.data)[:,0]*0
            n_iters = np.ceil(np.log(0.0000000001)/np.log((1-1/self.CDF.iloc[0]))).astype(int)
            self.benktander(exposure=exposure, apriori = 0, n_iters = n_iters,
                            triangle = triangle, inplace=True)
            self.method = 'DFM'
            return self
        if inplace==False:
            new_instance = copy.deepcopy(self)
            return new_instance.DFM(triangle=triangle.copy(), inplace=True)

    def born_ferg(self, exposure, apriori, triangle = None, inplace=False):
        """ Method to develop ultimates using the Bornheutter-Ferguson Method (BF)

        Parameters:
            exposure: pandas.Series or numpy.array
                Specifies the exposures to be used in the BF calculation.  The length of the
                exposure object my be equal to the number of origin periods in the Triangle.
            apriori: number
                The apriori estimate to be used in the BF method.
            triangle: Triangle or None
                Set as None will apply the DFM to the triangle embedded in the Chainladder object.
                This can be overriden to apply the DFM method to a new triangle.
            inplace: bool
                Set to true will mutate the Chainladder object, set to false will create a copy.

         Returns:
            Chainladder object fit with the BF method.
        """
        if inplace == True:
            self.benktander(exposure=exposure, apriori = apriori, n_iters = 1,
                            triangle = triangle, inplace=True)
            self.method = 'born_ferg'
            return self
        if inplace == False:
            new_instance = copy.deepcopy(self)
            return new_instance.born_ferg(exposure=exposure, apriori = apriori,
                                          triangle=triangle, inplace=True)

    def benktander(self, exposure, apriori, n_iters = 1, triangle = None, inplace=False):
        '''This method generalizes both the DFM and Bornheuetter-Ferguson (BF)
           methods.  It serves as the computational base for these other two methods
           by setting the n_iters parameter to 1 in the case of the BF method and
           a sufficiently high n_iters for the chainladder.
           L\sum_{k=0}^{n-1}(1-\frac{1}{CDF}) + Apriori\times (1-\frac{1}{CDF})^{n}
        '''
        if inplace == True:
            if triangle is None:
                triangle = self.triangle
            exposure = Series(exposure)
            exposure.index = triangle.data.index
            my_apriori = np.array(exposure) * apriori
            latest = np.array(triangle.get_latest_diagonal().iloc[:,-1])
            CDF = np.array(self.CDF)[::-1]
            CDF = np.expand_dims(np.array(CDF),1)
            CDF = 1-1/np.repeat(CDF,n_iters+1,axis=1)
            exponents = np.expand_dims(list(range(n_iters+1)),0)
            exponents = np.repeat(exponents,CDF.shape[0], axis=0)
            CDF = CDF**exponents
            self.ultimates = Series(np.sum(CDF[:,:-1],axis=1)*latest+CDF[:,-1]*my_apriori,
                                    index=triangle.data.index)
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
            exposure = Series(exposure)
            exposure.index = triangle.data.index
            latest = np.array(triangle.get_latest_diagonal().iloc[:,-1])
            latest = latest[exposure>0]
            CDF = np.array(self.CDF)[::-1]
            reported_exposure = np.array(exposure[exposure>0])/ CDF
            trend_array = np.array([(1+trend)**(sum(exposure>0) - (i+1)) for i in range(sum(exposure>0))])
            decay_matrix = np.array([[decay**abs(i-j) for i in range(sum(exposure>0) )] for j in range(sum(exposure>0))])
            weighted_exposure = reported_exposure * decay_matrix
            trended_ultimate = np.repeat(np.array([(latest * trend_array) /(reported_exposure)]),sum(exposure>0),axis=0)
            apriori = np.sum(weighted_exposure*trended_ultimate,axis=1)/np.sum(weighted_exposure,axis=1)
            detrended_ultimate = apriori/trend_array
            IBNR = detrended_ultimate * (1-1/CDF) * np.array(exposure[exposure>0])
            self.ultimates = Series(latest + IBNR, index=exposure[exposure>0].index)
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

    def inverse_power_tail(self):
        '''Calculates the tail using the inverse power curve fit
        '''

        X = np.array([np.log(1/(item+1)) for item in range(len(self.age_to_age().columns))])
        y = np.array(np.log(self.LDF-1))
        X = sm.add_constant(X)
        OLS = sm.OLS(y,X).fit()
        new_X = np.array(range(int(np.max(1/np.exp(X))),100000))
        self.tail_factor = np.product(1 + np.exp(np.log(1/new_X)*OLS.params[1]+OLS.params[0]))
        self.set_LDF(self.LDF_average, inplace=True)
        return self

    def full_triangle(self):
        '''Produces a 'completed' triangle.  Useful for tracking actuals vs
           expected over time. '''
        CDF = np.expand_dims(np.append(np.array(self.CDF),[1.],0),axis=0)
        CDF = np.repeat(CDF,self.triangle.data.shape[0],axis=0)
        ultimates = np.expand_dims(np.array(self.ultimates),axis=1)
        ultimates = np.repeat(ultimates,CDF.shape[1],axis=1)
        latest = np.append(np.array(self.triangle.data),
                           np.repeat(np.array([[np.nan]]),
                                     self.triangle.data.shape[0],
                                     axis=0),
                           axis=1)
        existing_elements_ind = np.nan_to_num(latest-latest+1)
        full = (np.nan_to_num(latest)*existing_elements_ind + \
                ultimates / CDF * (1-existing_elements_ind))
        return DataFrame(full,
                         index = self.triangle.data.index,
                         columns = list(self.triangle.data.columns) + ['Ultimate'])

    def get_expectation(self, num_periods, decay):
        '''Determines expected emergence from valuation date for a number of periods'''
        val_date = self.triangle.data.index.max()
        my_dict = {'Y':12,'Q':3,'M':1}
        num_months = num_periods*my_dict[self.triangle.development_grain]
        num_years = np.floor(num_months/12)
        num_months =  num_months % 12
        num_years = num_years + 1 if val_date.month+num_months > 12 else num_years
        num_months = num_months - 12  if val_date.month+num_months > 12 else num_months
        end_date = to_datetime(str(int(val_date.year+num_years)) + '-' + str(int(num_months + val_date.month)) + '-01').strftime('%Y-%m')
        val_date = val_date.strftime('%Y-%m')
        full = self.full_triangle().iloc[:,:-1]
        tail_ldf_initial = (1-decay)*(self.tail_factor-1)
        tail_decay = 1 + np.repeat(np.repeat(np.array([[tail_ldf_initial]]),full.shape[0],axis=0),240,axis=1) * \
                         decay**np.repeat(np.expand_dims(np.array(range(0,240)),0),full.shape[0],axis=0)
        tail_decay = np.cumprod(tail_decay, axis=1)*self.tail_factor/np.prod(tail_decay, axis=1)[0]
        tail_decay = np.repeat(np.expand_dims(np.array(full.iloc[:,-1]),1),tail_decay.shape[1],axis=1)*tail_decay
        full = np.append(full,tail_decay,axis=1)
        temp = DataFrame(full, index=self.full_triangle().index, columns = [item + 1 for item in range(full.shape[1])])
        temp = Triangle(temp).data_as_table().set_index('origin')
        return temp[temp['development']==end_date]['values'] - temp[temp['development']==val_date]['values']
