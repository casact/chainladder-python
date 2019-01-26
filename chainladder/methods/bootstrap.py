"""
Bootstrap Chainladder
=====================
"""
from sklearn.base import BaseEstimator
import numpy as np

class BootstrapChainladder(BaseEstimator):
    def __init__(self, n_sims = 999, process_distr="od poisson"):
        self.n_sims = n_sims
        self.process_distr = process_distr
        self.average = average

    def fit(X, y=None, sample_weight=None):
        self.weights = weights
        self.alpha = alpha
        self.tri = Chainladder(tri).triangle
        self.process_distr = process_distr
        sim_claims, IBNR, sim_LDF, scale_phi = self._get_simulation(X)
        self.IBNR = IBNR
        self.sim_claims = sim_claims
        self.sim_LDF = sim_LDF
        self.scale_phi = scale_phi
        self.avLDF = pd.Series(np.append(np.mean(self.sim_LDF,axis=0),[1]), index=Chainladder(tri).age_to_age().columns)

    def _get_simulation(self, X):
        raa = cl.load_dataset('raa')
        X = cl.Chainladder().fit(raa)
        exp_incr_triangle = X.full_expectation_.cum_to_incr().triangle[0,0, :, :-1]*X.X_.nan_triangle()
        unscaled_residuals = ((X.X_.cum_to_incr().triangle - exp_incr_triangle)/np.sqrt(np.abs(exp_incr_triangle)))[0,0,:,:]

        ## Calculate the Pearson scale parameter
        nobs = np.nansum(X.X_.nan_triangle())
        scale_factor = (nobs - 2*X.X_.shape[-1]+1) #Degree of freedom adjustment (N-p)
        scale_factor = nobs
        scale_phi = sum(sum(np.nan_to_num(unscaled_residuals)**2))/scale_factor
        adj_resids = unscaled_residuals * np.sqrt(nobs/scale_factor)

        ## Sample incremental claims
        ## Resample the adjusted residuals with replacement, creating a new
        ## past triangle of residuals.
        k, v, o, d = X.X_.shape
        exp_clms = exp_incr_triangle[0,0]
        resids = np.reshape(adj_resids,(k,v,o*d))

        adj_resid_dist = resids[np.isfinite(resids)]  # Missing k,v dimensions
        # Suggestions from Using the ODP Bootstrap Model: A Practitioners Guide
        adj_resid_dist = adj_resid_dist[adj_resid_dist>0]
        adj_resid_dist = adj_resid_dist - np.mean(adj_resid_dist)

        a = np.array([np.random.choice(adj_resid_dist, size=exp_incr_triangle.shape,replace=True)*(exp_incr_triangle*0+1) for item in range(999)]).reshape(999, exp_incr_triangle.shape[0], exp_incr_triangle.shape[1])
        b = np.repeat(np.expand_dims(exp_incr_triangle,axis=0),999,axis=0)
        sim_claims = (a*np.sqrt(abs(b))+b).cumsum(axis=2)


        lower_diagonal = (X.X_.nan_triangle()*0)
        lower_diagonal[np.isnan(lower_diagonal)]=1
        cl.Chainladder().fit(cl.Development().fit_transform(raa)).full_expectation_.triangle[...,:-1]*lower_diagonal

        if self.process_distr == 'gamma':
                    process_triangle = np.nan_to_num(np.array([np.random.gamma(shape=abs(item/scale_phi),scale=scale_phi)*np.sign(np.nan_to_num(item))for item in sim_exp_incr_triangle]))
                if self.process_distr == 'od poisson':
                    process_triangle = np.nan_to_num(np.array([np.random.poisson(lam=abs(item))*np.sign(np.nan_to_num(item))for item in sim_exp_incr_triangle]))
                IBNR = process_triangle.cumsum(axis=2)[:,:,-1]
