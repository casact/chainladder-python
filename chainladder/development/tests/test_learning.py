import chainladder as cl
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from chainladder.utils.utility_functions import PatsyFormula

def test_basic_odp_cl(genins):
    model = cl.DevelopmentML(Pipeline(steps=[
        ('design_matrix', PatsyFormula('C(development)')),
        ('model', LinearRegression(fit_intercept=False))]),
                y_ml=response,fit_incrementals=False).fit(genins)
    assert abs(model.triangle_ml_.loc[:,:,'2010',:] - genins.mean()).max() < 1e2