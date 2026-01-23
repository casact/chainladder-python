import chainladder as cl
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from chainladder.utils.utility_functions import PatsyFormula
import pandas as pd

def test_incremental(genins):
    response = [genins.columns[0]]
    model = cl.DevelopmentML(Pipeline(steps=[
        ('design_matrix', PatsyFormula('C(development)')),
        ('model', LinearRegression(fit_intercept=False))]),
                y_ml=response,fit_incrementals=False).fit(genins)
    assert abs(model.triangle_ml_.loc[:,:,'2010',:] - genins.mean()).max() < 1e2

def test_misc(genins):
    model = cl.DevelopmentML(Pipeline(steps=[
        ('design_matrix', PatsyFormula('C(development)')),
        ('model', LinearRegression(fit_intercept=False))]),
                weighted_step = ['model'], fit_incrementals=False).fit(genins, sample_weight=genins/genins)
    assert abs(model.triangle_ml_.loc[:,:,'2010',:] - genins.mean()).max() < 1e2
    
def test_grain():   
    grains={'Y':2,'2Q':4,'Q':8,'M':24}
    for period in grains.keys():
        tframe = pd.DataFrame()
        devdates=pd.date_range(start='2000-01-01',end='2002-01-01',freq=period+'S',inclusive='left')
        for ind, sdate in enumerate(devdates):
            tframe=pd.concat([tframe,pd.DataFrame({'origin':[sdate]*(len(devdates)-ind),
            'development':pd.date_range(start=sdate,end='2002-01-01',freq=period+'S',inclusive='left'),
            'loss':[100]*(len(devdates)-ind)})])
        tri= cl.Triangle(tframe,origin='origin',development='development',columns='loss',cumulative=False)
        model = cl.DevelopmentML(Pipeline(steps=[
        ('design_matrix', PatsyFormula('C(development)')),
        ('model', LinearRegression(fit_intercept=False))]),
                weighted_step = ['model'], fit_incrementals=False).fit(tri)
        assert grains[period] == len(model.triangle_ml_.development) 
        assert grains[period] == len(model.triangle_ml_.origin)