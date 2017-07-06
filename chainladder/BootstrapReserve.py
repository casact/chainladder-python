from chainladder.UtilityFunctions import load_dataset, Plot
from chainladder.Triangle import Triangle
from warnings import warn
from chainladder.Chainladder import Chainladder
import numpy as np
import pandas as pd





tri = Triangle(load_dataset('RAA'))


#if len(tri.data) != len(tri.data.columns): 
#            warn('MunichChainLadder does not support triangles with fewer development periods than origin periods.')


class BootChainladder:
    def __init__(self, tri, n_sims = 999, process_distr="gamma"):
        self.n_sims = n_sims
        self.tri = Chainladder(tri).triangle
        self.process_distr = process_distr
        sim_claims, IBNR, sim_LDF, scale_phi = self.__get_simulation()
        self.IBNR = IBNR
        self.sim_claims = sim_claims
        self.sim_LDF = sim_LDF
        self.scale_phi = scale_phi
        self.avLDF = pd.Series(np.append(np.mean(self.sim_LDF,axis=0),[1]), index=Chainladder(tri).age_to_age().columns)
   
    def __get_simulation(self):
        ## Obtain cumulative fitted values for the past triangle by backwards
        ## recursion, starting with the observed cumulative paid to date in the latest
        ## diagonal
        tri = self.tri
        n_sims = self.n_sims
        ults = np.repeat(np.array(Chainladder(tri).full_triangle.iloc[:,-1]),tri.ncol).reshape(len(tri.data),tri.ncol)
        CDF = np.array(list(Chainladder(tri).CDF)*len(Chainladder(tri).full_triangle)).reshape(len(Chainladder(tri).full_triangle),len(Chainladder(tri).CDF))
        exp_incr_triangle = Triangle(pd.DataFrame(ults/CDF, index=tri.data.index, columns=tri.data.columns)).cum_to_incr()*(tri.data*0+1)
        unscaled_residuals = (tri.cum_to_incr()-exp_incr_triangle)/np.sqrt(abs(exp_incr_triangle))
        
        ## Calculate the Pearson scale parameter
        nobs = 0.5 * tri.ncol * (tri.ncol + 1)
        scale_factor = (nobs - 2*tri.ncol+1)
        scale_phi = sum(sum(np.array(unscaled_residuals.replace(np.nan,0)**2)))/scale_factor
        adj_resids = unscaled_residuals * np.sqrt(nobs/scale_factor)
        
        ## Sample incremental claims
        ## Resample the adjusted residuals with replacement, creating a new
        ## past triangle of residuals.
        exp_clms = np.array(exp_incr_triangle)
        resids = np.array(adj_resids).flatten()   
        adj_resid_dist = resids[np.isfinite(resids)]
        a = np.array([np.random.choice(adj_resid_dist, size=exp_clms.shape,replace=True)*(exp_clms*0+1) for item in range(self.n_sims)]).reshape(self.n_sims, exp_clms.shape[0], exp_clms.shape[1])
        b = np.repeat(np.expand_dims(exp_clms,axis=0),self.n_sims,axis=0)
        sim_claims = (a*np.sqrt(abs(b))+b).cumsum(axis=2)
    
        sim_LDF = np.sum(np.nan_to_num(sim_claims[:,:,1:]),axis=1) / np.sum(np.nan_to_num((sim_claims[:,:,1:]*0+1)*sim_claims[:,:,:-1]),axis=1)
        sim_CDF = np.append(np.cumprod(sim_LDF[:,::-1],axis=1)[:,::-1],np.array([1]*n_sims).reshape(n_sims,1),axis=1)
        sim_latest = [np.flipud(np.array(item)).diagonal() for item in sim_claims] 
        sim_latest = np.flip(sim_claims,axis=1).diagonal(axis1=1,axis2=2)   
        
        sim_ults = (sim_latest*sim_CDF)
        lu = list(sim_ults)
        lc = list(sim_CDF)
        ## Get expected future claims
        ## Obtain the corresponding future triangle of incremental payments by
        ## differencing, to be used as the mean when simulating from the process
        ## distribution.
        sim_exp_cum_triangle = np.array([np.flipud(lu[num].reshape(tri.ncol,1).dot(1/lc[num].reshape(1,tri.ncol))) for num in range(n_sims)])
        sim_exp_incr_triangle = np.append(sim_exp_cum_triangle[:,:,0,np.newaxis],np.diff(sim_exp_cum_triangle),axis=2)
        sim_exp_incr_triangle = np.nan_to_num(sim_exp_incr_triangle * np.repeat(np.expand_dims(np.array((tri.data*0).replace(np.nan,1).replace(0,np.nan)),axis=0),n_sims,axis=0))
        # process for "Gamma"
        if self.process_distr == 'gamma':
            process_triangle = np.nan_to_num(np.array([np.random.gamma(shape=abs(item/scale_phi),scale=scale_phi)*np.sign(np.nan_to_num(item))for item in sim_exp_incr_triangle]))
        if self.process_distr == 'od poisson':
            process_triangle = np.nan_to_num(np.array([np.random.poisson(lam=abs(item))*np.sign(np.nan_to_num(item))for item in sim_exp_incr_triangle]))
        IBNR = process_triangle.cumsum(axis=2)[:,:,-1]
        return sim_claims, IBNR, sim_LDF, scale_phi


    def summary(self):
        IBNR = self.IBNR
        tri = self.tri
        summary = pd.DataFrame()
        summary['Latest'] = tri.get_latest_diagonal()
        summary['Mean Ultimate'] = summary['Latest'] + pd.Series([np.mean(np.array(IBNR)[:,num]) for num in range(len(tri.data))],index=summary.index)
        summary['Mean IBNR'] = summary['Mean Ultimate'] - summary['Latest'] 
        summary['SD IBNR'] = pd.Series([np.std(np.array(IBNR)[:,num]) for num in range(len(tri.data))],index=summary.index)
        summary['IBNR 75%'] = pd.Series([np.percentile(np.array(IBNR)[:,num],q=75) for num in range(len(tri.data))],index=summary.index)
        summary['IBNR 95%'] = pd.Series([np.percentile(np.array(IBNR)[:,num],q=95) for num in range(len(tri.data))],index=summary.index)
        return summary
    
    def plot(self, plots=['IBNR']): 
        """ Method, callable by end-user that renders the matplotlib plots.
        
        Arguments:
            plots: list[str]
                A list of strings representing the charts the end user would like
                to see.  If ommitted, all plots are displayed.  Available plots include:
                    ============== =================================================
                    Str            Description
                    ============== =================================================
                    IBNR           Bar chart with IBNR distribution
                    ============== =================================================
                    
        Returns:
            Renders the matplotlib plots.
            
        """   
        my_dict = []
        plot_dict = self.__get_plot_dict()
        for item in plots:
            my_dict.append(plot_dict[item])
        Plot(my_dict)
        
    def __get_plot_dict(self):
        IBNR = np.sum(self.IBNR,axis=1)
        plot_dict = {'IBNR':{'Title':'Histogram of total IBNR',
                                     'XLabel':'Total IBNR',
                                     'YLabel':'Frequency',
                                     'chart_type_dict':{'type':['hist'],
                                                       'x':[IBNR],
                                                       'bins':[min(int(self.n_sims/50),250)]
                                                       }}
                    }
        return plot_dict