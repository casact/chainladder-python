# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 07:09:27 2017

@author: jboga
"""
from Triangles import triangle
from numpy import sqrt

class chainladder:
    def __init__(self, tri, weights=1, delta=1):
        self.triangle = tri
        weights = self.__checkWeights(weights)
        delta = [delta for item in range(len(self.triangle.data.columns)-1)]
        self.delta = delta
        self.weights = triangle(weights)     
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

def get_tail(tri):
    return Series([1], index = [tri.data.iloc[:,-1].name + '-Ult'])

def get_ldf(tri, alpha=0, w=1):
    ## Mack uses alpha between 0 and 2 to distinguish:
    ## alpha = 0 straight averages of LDFs
    ## alpha = 1 historical chain ladder Volume weighted age-to-age factors
    ## alpha = 2 ordinary regression with intercept 0
    lm = LinearRegression(fit_intercept=False)
    ldf = Series()
    for i in range(0, len(tri.data.columns)-1):
        data = tri.data.iloc[:,i:i+2].dropna()
        X = data.iloc[:,0].values
        y = data.iloc[:,1].values
        lm.fit(X.reshape(-1, 1),y, sample_weight=w/X**(2-alpha))
        ldf = ldf.append(Series(lm.coef_))
    ldf.index = ata(tri).columns
    ldf = ldf.append(get_tail(tri))
    ldf.name = 'ldf'
    return ldf

   
def get_cdf(tri, alpha=0):
    ldf = get_ldf(tri, alpha)
    cdf = ldf[::-1].cumprod().iloc[::-1]
    return cdf

def get_latest_diag(tri):
    latest = DataFrame([[int(tri.data.iloc[0].dropna().index[-1]), tri.data.iloc[0].dropna()[-1]]], columns=['dev','values'])
    for i in range(len(tri.data)-1):
        latest = latest.append(DataFrame([[int(tri.data.iloc[i+1].dropna().index[-1]), tri.data.iloc[i+1].dropna()[-1]]], columns=['dev','values']))
    latest.index=[tri.data.index]
    return latest
        
def get_ult(tri, alpha = 0):
    cdf_iloc = [int(item[:item.find('-')]) for item in get_cdf(tri, alpha).index.values]
    latest = get_latest_diag(tri)
    cdf = get_cdf(tri, alpha)
    relevant_cdf = cdf.iloc[[cdf_iloc.index(item) for item in latest['dev']]]
    latest['CDF'] = relevant_cdf.values
    latest['Ultimate'] = latest['values'].values * relevant_cdf.values
    latest.rename(columns={'values':'Latest'}, inplace=True)
    return latest.drop(['dev'], axis=1)


#   Weighted Regression Through Origin
class WRTO:
    def __init__(self,X,y,w):
        # I could not find any Python package that does Weighted regression through origin
        # that also produces summary statistics, so I wrote my own...
        self.X = X
        self.y = y
        self.w = w
        self.coef_ = sum(w*X*y)/sum(w*X*X)
        self.WSSResidual = sum(w*((y-self.coef_*X)**2))
        self.mse = self.WSSResidual / (len(X)-1)
        self.se = sqrt(self.mse/sum(w*X*X))
        self.sigma = sqrt(self.mse)
        



    