# -*- coding: utf-8 -*-

"""The Classes module.

The :mod:`Classes` module contains three classes:

- :class:`Classes.Triangle`
- :class:`Classes.ChainLadder`
- :class:`Classes.WRTO`
- :class:`Classes.MackChainLadder`

One can use the :func:`Classes.Triangle.incr2cum` and
:func:`Classes.Triangle.cum2incr` functions to accumulate and decumulate a triangle.

   
"""
import numpy as np
from pandas import DataFrame, concat, Series, pivot_table


class Triangle:
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
        self.latest_values = Series([row.dropna().iloc[-1] for index, row in self.data.iterrows()], index=self.data.index)
          
    def dataAsTable(self, inplace=False):
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
            return self.data
        

    def dataAsTriangle(self, inplace=False):
        """Method to convert tabular form to triangle form.

        Arguments:
            inplace: bool
                Set to True will update the instance data attribute inplace 

        Returns:
            Updated instance `data` parameter if inplace is set to True otherwise it returns a pandas.DataFrame
        """
        if self.dataform == 'tabular':
            tri = pivot_table(self.data,values=self.values,index=[self.origin], columns=[self.dev]).sort_index()
            tri.columns = [str(item) for item in tri.columns]
            if inplace == True:
                self.data = tri   
                self.dataform = 'triangle'
            return tri
        else:
            return self.data
        
    def incr2cum(self, inplace=False):
        """Method to convert an incremental triangle into a cumulative triangle.  Note,
        the triangle must be in triangle form.

        Arguments:
            inplace: bool
                Set to True will update the instance data attribute inplace 

        Returns:
            Updated instance `data` parameter if inplace is set to True otherwise it returns a pandas.DataFrame
        
        todo:
            Need to error check dataform and convert it to triangle form before running method.
        """
        incr = DataFrame(self.data.iloc[:,0])
        for val in range(1, len(self.data.T.index)):
            incr = concat([incr,self.data.iloc[:,val]+incr.iloc[:,-1]],axis=1)
        incr = incr.rename_axis('dev', axis='columns')
        incr.columns = self.data.T.index
        if inplace == True:
            self.data = incr
        return incr     
    
    def cum2incr(self, inplace=False):
        """Method to convert an cumulative triangle into a incremental triangle.  Note,
        the triangle must be in triangle form.

        Arguments:
            inplace: bool
                Set to True will update the instance data attribute inplace 

        Returns:
            Updated instance `data` parameter if inplace is set to True otherwise it returns a pandas.DataFrame
        
        TODOs:
            Need to error check dataform and convert it to triangle form before running method.
        """
        incr = self.data.iloc[:,0]
        for val in range(1, len(self.data.T.index)):
            incr = concat([incr,self.data.iloc[:,val]-self.data.iloc[:,val-1]],axis=1)
        incr = incr.rename_axis('dev', axis='columns')
        incr.columns = self.data.T.index
        if inplace == True:
            self.data = incr        
        return incr   
    
    def __set_origin(self):
        """Experimental hidden method. Purpose is to profile the data and autodetect the origin period
        improving the user experience by not requiring the user to supply an explicit origin.
        
        TODOs:
            1. Continually refine potential origin_names to a broader list
            2. Need error handling
        """
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
        """Experimental hidden method. Purpose is to profile the data and autodetect the dev period
        improving the user experience by not requiring the user to supply an explicit dev.
        
        TODOs:
            1. Continually refine potential dev_names to a broader list
            2. Need error handling
        """
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
        """Experimental hidden method. Purpose is to profile the data and autodetect the values parameter
        improving the user experience by not requiring the user to supply an explicit values parameter.
        This is onyl necessary when dataform is 'tabular'.
        
        TODOs:
            1. Continually refine potential values_names to a broader list
            2. Need error handling
        """
        ##### Identify dev Profile ####
        value_names = ['incurred claims'] 
        values_match = [i for i in value_names if i in self.data.columns]
        if len(values_match)==1:
            self.values = values_match[0]
        else:
            self.values = 'values'
        return 
    
class WRTO:
    """Weighted least squares regression through the origin

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
        if len(X) == 1:
            self.mse = np.nan
        else:
            self.mse = self.WSSResidual / (len(X)-1)
        self.se = np.sqrt(self.mse/sum(w*X*X))
        self.sigma = np.sqrt(self.mse)
        

class ChainLadder:
    """ ChainLadder class specifies the chainladder model.
    
    The classical chain-ladder is a deterministic algorithm to forecast claims based on
    historical data. It assumes that the proportional developments of claims from one
    development period to the next are the same for all origin years.
    `Proper Citation Needed... <https://github.com/mages/ChainLadder>`_
    
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
            A set of values representing an input into the weights of the WRTO class.
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
        """ Hidden method used to convert weights from a scalar into a matrix of the 
        same shape as the triangle
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
            sample_weight=w/X**self.delta[i]
            #lm.fit(X.reshape(-1, 1),y, sample_weight=w/X**self.delta[i])
            lm = WRTO(X,y,sample_weight)
            models.append(lm)   
        return models
    
    def ata(self, colname_sep = '-'):
        """ Method to display an age-to-age triangle with a display of simple average
        chainladder development factors and volume weighted average development factors.
        
        Parameters:    
            colname_sep : str
                text to join the names of two adjacent columns representing the age-to-age factor column name.
        
        Returns:
            Pandas.DataFrame of the age-to-age triangle.
        """
        incr = DataFrame(self.triangle.data.iloc[:,1]/self.triangle.data.iloc[:,0])
        for i in range(1, len(self.triangle.data.columns)-1):
            incr = concat([incr,self.triangle.data.iloc[:,i+1]/self.triangle.data.iloc[:,i]],axis=1)
        incr.columns = [item + colname_sep + self.triangle.data.columns.values[num+1] for num, item in enumerate(self.triangle.data.columns.values[:-1])]
        incr = incr.iloc[:-1]
        ldf = [item.coef_ for item in ChainLadder(self.triangle, delta=2).models]
        incr.loc['smpl']=ldf
        ldf = [item.coef_ for item in ChainLadder(self.triangle, delta=1).models]
        incr.loc['vwtd']=ldf
        return incr.round(3)
    


class MackChainLadder:
    """ MackChainLadder class specifies the Mack chainladder model.
    
    Thomas Mack published in 1993 [Mac93] a method which estimates the standard
    errors of the chain-ladder forecast without assuming a distribution under three
    conditions.
    `Proper Citation Needed... <https://github.com/mages/ChainLadder>`_
    
    Parameters:    
        tri : Triangle
            A triangle object. Refer to :class:`Classes.Triangle`
        weights : int
            A value representing an input into the weights of the WRTO class.
        alpha : int
            A value representing an input into the weights of the WRTO class.
        tail : bool
            Represent whether a tail factor should be applied to the data. Value of False
            sets tail factor to 1.0
        
    Attributes:
        tri : Triangle
            A triangle object. Refer to :class:`Classes.Triangle`
        weights : pandas.DataFrame
            A value representing an input into the weights of the WRTO class.
        fullTriangle : pandas.DataFrame
            A completed triangle using Mack Chainladder assumptions.
        f : numpy.array
            An array representing the (col-1) loss development factors, f-notation borrowed from Mack
        fse : numpy.array
            An array representing the (col-1) standard errors of loss development factors.
    
    """
    
    def __init__(self, tri,  weights=1,  alpha=1,  est_sigma="log-linear",  
                 tail=False,  tail_se=None,  tail_sigma=None,  mse_method = "Mack"):
        delta = 2-alpha
        if tail == False:
            self.tail_factor = 1.0
        else:
            self.tail_factor = 2.0
        self.triangle = tri
        self.triangle.dataAsTriangle(inplace=True)    
        cl = ChainLadder(self.triangle, weights=weights,  delta=delta)
        alpha = [2 - item for item in cl.delta]
        
        
        self.weights = cl.weights
        self.fullTriangle = cl.predict()
        self.alpha = alpha
        self.f = np.append(np.array([item.coef_ for item in cl.models]), self.tail_factor)
        self.fse = np.array([item.se for item in cl.models])[:-1]
        self.sigma = np.array([item.sigma for item in cl.models])[:-1]
        self.sigma = np.append(self.sigma,self.tail_sigma())
        self.fse = np.append(self.fse,self.tail_se())
        self.Fse = self.Fse()
        self.total_process_risk = np.sqrt((self.process_risk()**2).sum())
        self.total_parameter_risk = self.total_parameter_risk()
        self.mack_se = np.sqrt(self.process_risk()**2 + self.parameter_risk()**2)
        self.total_mack_se = np.sqrt(self.total_process_risk[-1]**2+self.total_parameter_risk[-1]**2)
        
        


    def process_risk(self):
        """ Method to return the process risk of the Mack Chainladder model.
        
        Returns:
            
        """
        procrisk = DataFrame([0 for item in range(len(self.fullTriangle))],index=self.fullTriangle.index, columns=[self.fullTriangle.columns[0]])
        for i in range(1,len(self.fullTriangle.columns)):
            temp = DataFrame(np.sqrt((self.fullTriangle.iloc[:,i-1]*self.Fse.iloc[:,i-1])**2 + (self.f[i-1]*procrisk.iloc[:,i-1])**2)*self.triangle.data.iloc[:,i].isnull())
            temp.columns = [self.fullTriangle.columns[i]]
            procrisk = concat([procrisk, temp],axis=1)
        return procrisk

    def parameter_risk(self):
        paramrisk = DataFrame([0 for item in range(len(self.fullTriangle))],index=self.fullTriangle.index, columns=[self.fullTriangle.columns[0]])
        for i in range(1,len(self.fullTriangle.columns)):
            temp = DataFrame(np.sqrt((self.fullTriangle.iloc[:,i-1]*self.fse[i-1])**2 + (self.f[i-1]*paramrisk.iloc[:,i-1])**2)*self.triangle.data.iloc[:,i].isnull())
            temp.columns = [self.fullTriangle.columns[i]]
            paramrisk = concat([paramrisk, temp],axis=1)
        return paramrisk
    
    def total_parameter_risk(self):
        M = np.empty(0)
        tpr = [0]
        for i in range(len(self.fullTriangle.columns)):
            M = np.append(M, np.array(sum(self.fullTriangle.iloc[:,i].iloc[-(i+1):])))
        for i in range(len(self.fullTriangle.columns)-1):
            tpr.append(np.sqrt((M[i]*self.fse[i])**2 + (tpr[-1]*self.f[i])**2))
        return np.array(tpr)    
            
        
    def Mack_SE(self):
        return  DataFrame(np.sqrt(np.matrix(self.process_risk()**2 )+np.matrix(self.parameter_risk()**2)), index=self.fullTriangle.index)

    def Fse(self):
        # This is sloppy, and I don't know that it works for all cases.  Need to
        # understand weights better.
        fulltriangleweightconst = self.weights.data.mode().T.mode().iloc[0,0]
        fulltriangleweight = self.fullTriangle*0 + fulltriangleweightconst
        Fse = DataFrame()
        for i in range(self.fullTriangle.shape[1]-1):
            Fse = concat([Fse, DataFrame(self.sigma[i]/np.sqrt(np.array(fulltriangleweight.iloc[:,i]).astype(float)*np.array(self.fullTriangle.iloc[:,i]).astype(float)**self.alpha[i]))],axis=1)

        Fse.set_index(self.fullTriangle.index, inplace = True)
        return Fse
    
    def tail_se(self):
        if True:
            tailse = self.sigma[-1]/np.sqrt(self.fullTriangle.iloc[0,-2]**self.alpha[-1])
            
        else:
            # I cannot replicate R exactly!!!
            n = len(self.fullTriangle.columns)
            f = self.f[:-2]
            dev = np.array(self.fullTriangle.columns[:-2]).astype(int)
            ldf_reg = np.polyfit(dev, np.log(f-1),1)
            time_pd = (np.log(self.f[-2]-1)-ldf_reg[1])/ldf_reg[0]
            print(time_pd)
            fse = self.fse
            fse_reg = np.polyfit(dev, np.log(fse),1)
            tailse = np.exp(time_pd*fse_reg[0]+fse_reg[1])
        return tailse
    
    def tail_sigma(self):
        if True:
            y = np.log(self.sigma)
            x = np.array([i for i in range(len(self.sigma))])
            model = np.polyfit(x,y,1)
            tailsigma = np.exp((x[-1]+1)*model[0]+model[1])
            #if self.est_sigma == 'Mack':
            #    np.sqrt(abs(min((y[-1]**4/y[-2]**2),min(y[-2]**2, y[-1]**2))))
                
        else:
            # I cannot replicate R exactly!!!
            n = len(self.fullTriangle.columns)
            f = self.f[:-2]
            dev = np.array(self.fullTriangle.columns[:-2]).astype(int)
            ldf_reg = np.polyfit(dev, np.log(f-1),1)
            time_pd = (np.log(self.f[-2]-1)-ldf_reg[1])/ldf_reg[0]
            sigma = self.sigma
            sigma_reg = np.polyfit(dev, np.log(sigma),1)
            tailsigma = np.exp(time_pd*sigma_reg[0]+sigma_reg[1])
        return tailsigma   
    
    def summary(self):
        summary = DataFrame()
        summary['Latest'] = Series([row.dropna().iloc[-1] for index, row in self.triangle.data.iterrows()], index=self.triangle.data.index)      
        summary['Dev to Date'] = Series([row.dropna().iloc[-1] for index, row in self.triangle.data.iterrows()], index=self.triangle.data.index)/self.fullTriangle.iloc[:,-1]        
        summary['Ultimate'] = self.fullTriangle.iloc[:,-1]  
        summary['IBNR'] = summary['Ultimate']-summary['Latest']
        summary['Mack S.E.'] = self.mack_se.iloc[:,-1]
        summary['CV(IBNR)'] = summary['Mack S.E.']/summary['IBNR']
        return summary
    
        
        
        

      