"""The BootsrapReserver module and BootChainladder class establishes the
bootstrap method of the volume-weighted chainladder method.  A good reference
is `[Shapland16] <citations.html>`_.
NOT IMPLEMENTED YET...
"""

import numpy as np
import copy
from chainladder.methods.base import MethodBase


class BootstrapCL(MethodBase):
    """ BootChainladder implements the Overdispersed Poisson Bootstrap
    Chainladder model.

    Parameters:
        tri : `Triangle <Triangle.html>`_
            A triangle object. Refer to :class:`Classes.Triangle`
        n_sims : int
            A value representing the number of simulations to run.
        process_distr : str
            A value representing the bootstrap residual distribuion structure.


    Attributes:
        tri : `Triangle <Triangle.html>`_
            A triangle object. Refer to :class:`Classes.Triangle`
        n_sims : int
            A value representing the number of simulations to run.
        process_distr : str
            A value representing the bootstrap residual distribuion structure.
        IBNR: numpy.array
            An array with shape (n_sims, nrow, ncol) representing the IBNR of
            the simulated triangles.  ncol is an attribute of the `Triangle` class,
            and nrow is the row length of the triangle.
        sim_claims: nupmy.array
             An array with shape (n_sims, nrow, ncol) representing the simulated
            incremental loss amounts sampled from the Pearson scaled residuals.
        sim_LDF: numpy.array
            An array with shape (n_sims, ncol) representing the volume weighted
            LDFs for each simluation.
        scale_phi: float32
            A value representing the Variance scale parameter in the GLM framework.
        avLDF: numpy.array
            The implied LDFs from the mean IBNR of the simulated IBNR values.

    """
    def __init__(self,n_sims = 999, process_distr="od poisson"):
        self.n_sims = n_sims
        self.process_distr = process_distr

    def fit(self, X, y=None, sample_weight=None):
        self.X_ = self.validate_X(X)
        sim_claims, IBNR, sim_LDF, scale_phi = self._get_simulation()
        self.IBNR = IBNR
        self.sim_claims = sim_claims
        self.sim_LDF = sim_LDF
        self.scale_phi = scale_phi
        self.avLDF = pd.Series(np.append(np.mean(self.sim_LDF,axis=0),[1]), index=Chainladder(tri).age_to_age().columns)


    def _get_simulation(self):
        ## Obtain cumulative fitted values for the past triangle by backwards
        ## recursion, starting with the observed cumulative paid to date in the latest
        ## diagonal
        tri = self.X_.triangle
        n_sims = self.n_sims
        LDF, CDF, ults, exp_incr_triangle = self._model_form(np.array([np.array(tri.data)]))
        exp_incr_triangle = exp_incr_triangle[0,:,:]*(tri.data*0+1)

        unscaled_residuals = (tri.cum_to_incr().data-exp_incr_triangle)/np.sqrt(abs(exp_incr_triangle))

        ## Calculate the Pearson scale parameter
        nobs = 0.5 * tri.ncol * (tri.ncol + 1) # This is N
        scale_factor = (nobs - 2*tri.ncol+1) #Degree of freedom adjustment (N-p)
        scale_factor = nobs
        scale_phi = sum(sum(np.array(unscaled_residuals.replace(np.nan,0)**2)))/scale_factor
        adj_resids = unscaled_residuals * np.sqrt(nobs/scale_factor)

        ## Sample incremental claims
        ## Resample the adjusted residuals with replacement, creating a new
        ## past triangle of residuals.
        exp_clms = np.array(exp_incr_triangle)
        resids = np.array(adj_resids).flatten()
        adj_resid_dist = resids[np.isfinite(resids)]
        # Suggestions from Using the ODP Bootstrap Model: A Practitioners Guide
        adj_resid_dist = adj_resid_dist[adj_resid_dist>0]
        adj_resid_dist = adj_resid_dist - np.mean(adj_resid_dist)

        a = np.array([np.random.choice(adj_resid_dist, size=exp_clms.shape,replace=True)*(exp_clms*0+1) for item in range(self.n_sims)]).reshape(self.n_sims, exp_clms.shape[0], exp_clms.shape[1])
        b = np.repeat(np.expand_dims(exp_clms,axis=0),self.n_sims,axis=0)
        sim_claims = (a*np.sqrt(abs(b))+b).cumsum(axis=2)

        sim_LDF, sim_CDF, sim_ults, sim_exp_incr_triangle = self._model_form(sim_claims)
        sim_exp_incr_triangle = np.nan_to_num(sim_exp_incr_triangle * np.repeat(np.expand_dims(np.array((tri.data*0).replace(np.nan,1).replace(0,np.nan)),axis=0),n_sims,axis=0))
        # process for "Gamma"
        if self.process_distr == 'gamma':
            process_triangle = np.nan_to_num(np.array([np.random.gamma(shape=abs(item/scale_phi),scale=scale_phi)*np.sign(np.nan_to_num(item))for item in sim_exp_incr_triangle]))
        if self.process_distr == 'od poisson':
            process_triangle = np.nan_to_num(np.array([np.random.poisson(lam=abs(item))*np.sign(np.nan_to_num(item))for item in sim_exp_incr_triangle]))
        IBNR = process_triangle.cumsum(axis=2)[:,:,-1]
        return sim_claims, IBNR, sim_LDF, scale_phi

    def _model_form(self, tri_array):
        ''' takes a multi-dimensional array and solves for cdf, ldf, and ultimate '''
        ''' very similar to development object, but we'd need to extend development object by n_sims '''

        w = np.nan_to_num(self.weights/tri_array[:,:,:-1]**(2-self.alpha))
        x = np.nan_to_num(tri_array[:,:,:-1]*(tri_array[:,:,1:]*0+1))
        y = np.nan_to_num(tri_array[:,:,1:])
        LDF = np.sum(w*x*y,axis=1)/np.sum(w*x*x,axis=1)
        # assumes no tail
        CDF = np.append(np.cumprod(LDF[:,::-1],axis=1)[:,::-1],np.array([1]*tri_array.shape[0]).reshape(tri_array.shape[0],1),axis=1)
        latest = np.flip(tri_array,axis=1).diagonal(axis1=1,axis2=2)
        ults = latest*CDF
        lu = list(ults)
        lc = list(CDF)
        exp_cum_triangle = np.array([np.flipud(lu[num].reshape(tri_array.shape[2],1).dot(1/lc[num].reshape(1,tri_array.shape[2]))) for num in range(tri_array.shape[0])])
        exp_incr_triangle = np.append(exp_cum_triangle[:,:,0,np.newaxis],np.diff(exp_cum_triangle),axis=2)
        return LDF, CDF, ults, exp_incr_triangle

    def summary(self):
        """ Method to produce a summary table of of the Mack Chainladder
        model.

        Returns:
            This calculation is consistent with the R calculation
            BootChainLadder$summary

        """
        IBNR = self.IBNR
        tri = self.tri
        summary = pd.DataFrame()
        summary['Latest'] = tri.get_latest_diagonal().iloc[:,-1]
        summary['Mean Ultimate'] = summary['Latest'] + pd.Series([np.mean(np.array(IBNR)[:,num]) for num in range(len(tri.data))],index=summary.index)
        summary['Mean IBNR'] = summary['Mean Ultimate'] - summary['Latest']
        summary['SD IBNR'] = pd.Series([np.std(np.array(IBNR)[:,num]) for num in range(len(tri.data))],index=summary.index)
        summary['IBNR 75%'] = pd.Series([np.percentile(np.array(IBNR)[:,num],q=75) for num in range(len(tri.data))],index=summary.index)
        summary['IBNR 95%'] = pd.Series([np.percentile(np.array(IBNR)[:,num],q=95) for num in range(len(tri.data))],index=summary.index)
        return summary
