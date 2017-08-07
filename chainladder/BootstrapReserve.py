"""The BootsrapReserver module and BootChainladder class establishes the 
bootstrap method of the volume-weighted chainladder method.  A good reference
is `[Shapland16] <citations.html>`_.

"""

from chainladder.UtilityFunctions import Plot, boxwhisker
from warnings import warn
from chainladder.Chainladder import Chainladder
import statsmodels.api as sm
import numpy as np
import pandas as pd

#if len(tri.data) != len(tri.data.columns): 
#            warn('MunichChainLadder does not support triangles with fewer development periods than origin periods.')

class BootChainladder:
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
    def __init__(self, tri, n_sims = 999, process_distr="od poisson"):
        weights = 1 
        alpha = 1
        self.n_sims = n_sims
        self.weights = weights
        self.alpha = alpha
        self.tri = Chainladder(tri).triangle
        self.process_distr = process_distr
        sim_claims, IBNR, sim_LDF, scale_phi = self.__get_simulation()
        self.IBNR = IBNR
        self.sim_claims = sim_claims
        self.sim_LDF = sim_LDF
        self.scale_phi = scale_phi
        self.avLDF = pd.Series(np.append(np.mean(self.sim_LDF,axis=0),[1]), index=Chainladder(tri).age_to_age().columns)
   
    def __repr__(self):   
        return str(self.summary())
    
    def __get_simulation(self):
        ## Obtain cumulative fitted values for the past triangle by backwards
        ## recursion, starting with the observed cumulative paid to date in the latest
        ## diagonal
        tri = self.tri
        n_sims = self.n_sims
        LDF, CDF, ults, exp_incr_triangle = self.__model_form(np.array([np.array(tri.data)]))
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
    
        sim_LDF, sim_CDF, sim_ults, sim_exp_incr_triangle = self.__model_form(sim_claims)
        sim_exp_incr_triangle = np.nan_to_num(sim_exp_incr_triangle * np.repeat(np.expand_dims(np.array((tri.data*0).replace(np.nan,1).replace(0,np.nan)),axis=0),n_sims,axis=0))
        # process for "Gamma"
        if self.process_distr == 'gamma':
            process_triangle = np.nan_to_num(np.array([np.random.gamma(shape=abs(item/scale_phi),scale=scale_phi)*np.sign(np.nan_to_num(item))for item in sim_exp_incr_triangle]))
        if self.process_distr == 'od poisson':
            process_triangle = np.nan_to_num(np.array([np.random.poisson(lam=abs(item))*np.sign(np.nan_to_num(item))for item in sim_exp_incr_triangle]))
        IBNR = process_triangle.cumsum(axis=2)[:,:,-1]
        return sim_claims, IBNR, sim_LDF, scale_phi
    
    def __model_form(self, tri_array):
        w = np.nan_to_num(self.weights/tri_array[:,:,:-1]**(2-self.alpha))
        x = np.nan_to_num(tri_array[:,:,:-1]*(tri_array[:,:,1:]*0+1))
        y = np.nan_to_num(tri_array[:,:,1:])
        LDF = np.sum(w*x*y,axis=1)/np.sum(w*x*x,axis=1)
        #Chainladder (alpha=1/delta=1)
        #LDF = np.sum(np.nan_to_num(tri_array[:,:,1:]),axis=1) / np.sum(np.nan_to_num((tri_array[:,:,1:]*0+1)*tri_array[:,:,:-1]),axis=1)
        #print(LDF.shape)
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
        summary['Latest'] = tri.get_latest_diagonal()
        summary['Mean Ultimate'] = summary['Latest'] + pd.Series([np.mean(np.array(IBNR)[:,num]) for num in range(len(tri.data))],index=summary.index)
        summary['Mean IBNR'] = summary['Mean Ultimate'] - summary['Latest'] 
        summary['SD IBNR'] = pd.Series([np.std(np.array(IBNR)[:,num]) for num in range(len(tri.data))],index=summary.index)
        summary['IBNR 75%'] = pd.Series([np.percentile(np.array(IBNR)[:,num],q=75) for num in range(len(tri.data))],index=summary.index)
        summary['IBNR 95%'] = pd.Series([np.percentile(np.array(IBNR)[:,num],q=95) for num in range(len(tri.data))],index=summary.index)
        return summary
    
    def plot(self, ctype='m', plots=['IBNR','ECDF', 'Ultimate','Latest'], plot_width=400, plot_height=275): 
        """ Method, callable by end-user that renders the matplotlib plots.
        
        Arguments:
            plots: list[str]
                A list of strings representing the charts the end user would like
                to see.  If ommitted, all plots are displayed.  Available plots include:
                    ============== ===================================================
                    Str            Description
                    ============== ===================================================
                    IBNR           Bar chart with IBNR distribution
                    ECDF           Expected Cumulative Distribution Function of IBNR
                    Ultimate       Box plot with mean ultimate value and variance
                    Latest         Box plot with latest incremental value and variance
                    ============== ===================================================
                    
        Returns:
            Renders the matplotlib plots.
            
        """   
        my_dict = []
        plot_dict = self.__get_plot_dict()
        for item in plots:
            my_dict.append(plot_dict[item])
        return Plot(ctype, my_dict, plot_width=plot_width, plot_height=plot_height).grid
        
    def __get_plot_dict(self):
        IBNR = np.sum(self.IBNR,axis=1)
        hist, edges = np.histogram(IBNR, density=True, bins=min(int(self.n_sims/50),250))
        kdest = sm.nonparametric.KDEUnivariate(IBNR)
        kdest.fit()
        plot_dict = {'IBNR':{'Title':'Histogram of total IBNR',
                                     'XLabel':'Total IBNR',
                                     'YLabel':'Frequency',
                                     'chart_type_dict':{'mtype':['hist'],
                                                       'x':[IBNR,kdest.support],
                                                       'bins':[min(int(self.n_sims/50),250)],
                                                       'label':['Total IBNR'],
                                                       
                                                       'type':['quad','line'],
                                                       'top':[hist,None],
                                                       'bottom':[[0]*min(int(self.n_sims/50),250),None],
                                                       'left':[edges[:-1],None],
                                                       'right':[edges[1:],None],
                                                       
                                                       'y':[None,pd.DataFrame(kdest.density, index=kdest.support).T],
                                                       'line_width':[None,2],
                                                       'line_cap':[None,'round'],
                                                       'line_join':[None,'round'],
                                                       'line_dash':[None,'solid'],
                                                       'alpha':[.5,.5,],
                                                       'label':['simluation',['kde']],
                                                       'rows':[None,1],
                                                       'color':['blue',['red']]
                                                       }},
                     'ECDF':{'Title':'ECDF(Total IBNR)',
                                     'XLabel':'Total IBNR',
                                     'YLabel':'F(x)',
                                     'chart_type_dict':{'mtype':['plot', 'plot'],
                                                       'x':[np.sort(np.sum(self.IBNR,axis=1)), np.sort(np.random.normal(np.mean(np.sum(self.IBNR,axis=1)),np.std(np.sum(self.IBNR,axis=1)),10000))],
                                                       'yM':[np.array(list(range(1,self.n_sims+1)))/self.n_sims, np.array(list(range(1,10000+1)))/10000],
                                                       'markerM':['.',''],
                                                       'linestyle':['-','-'],
                                                       'colorM':['red','blue'],
                                                       'alpha':[.5,1],
                                                       'type':['line','line'],
                                                       'y':[pd.DataFrame(np.array(list(range(1,self.n_sims+1)))/self.n_sims).T, pd.DataFrame(np.array(list(range(1,10000+1)))/10000).T],
                                                       'line_width':[3,2],
                                                       'line_cap':['round','round'],
                                                       'line_join':['round','round'],
                                                       'line_dash':['solid','dashed'],
                                                       'label':[['IBNR'],['Normal']],
                                                       'rows':[1,1],
                                                       'color':[['blue'],['red']]
                                                       }},
                     'Ultimate':{'Title':'Simulated Ultimate Claim Cost',
                                     'XLabel':'Origin Period',
                                     'YLabel':'Simulated Cost',
                                     'chart_type_dict':{'mtype':['plot','box'],
                                                       'x':[self.summary().index, np.repeat(np.expand_dims(np.array(self.summary()['Latest']),axis=0), self.n_sims,axis=0)+self.IBNR],
                                                       'positions':[None,self.summary().index],
                                                       'yM':[self.summary()['Mean Ultimate'],None],
                                                       'markerM':['o',None],
                                                       'linestyle':['',None],
                                                       'colorM':['red',None],
                                                       'alpha':[.5,None],
                                                       'label':['Mean Ultimate',None],
                                                       'type':['scatter','box'],
                                                       'y':[self.summary()['Mean Ultimate'], pd.DataFrame(np.repeat(np.expand_dims(np.array(self.summary()['Latest']),axis=0), self.n_sims,axis=0)+self.IBNR, columns = self.tri.data.index)],
                                                       'marker':['circle',None],
                                                       'color':['red',None]
                                                       }},
                     'Latest':{'Title':'Latest Actual Incremental Claims vs. Simulation',
                                     'XLabel':'Origin Period',
                                     'YLabel':'Simulated Latest',
                                     'chart_type_dict':{'mtype':['plot','box'],
                                                       'x':[self.summary().index, (np.flip(self.sim_claims,axis=1).diagonal(axis1=1,axis2=2) - np.append(np.array([[0]]*self.n_sims),np.flip(self.sim_claims[:,:-1,:-1],axis=1).diagonal(axis1=1,axis2=2),axis=1))[:,::-1]],
                                                       'positions':[None,self.summary().index],
                                                       'yM':[(np.diag(np.flip(np.array(self.tri.data),axis=0)) -np.append([0],np.diag(np.flip(np.array(self.tri.data.iloc[:-1,:-1]),axis=0))))[::-1],None],
                                                       'markerM':['o',None],
                                                       'linestyle':['',None],
                                                       'colorM':['red',None],
                                                       'alpha':[.5,None],
                                                       'label':['Latest Incremental',None],
                                                       'type':['scatter','box'],
                                                       'y':[(np.diag(np.flip(np.array(self.tri.data),axis=0)) -np.append([0],np.diag(np.flip(np.array(self.tri.data.iloc[:-1,:-1]),axis=0))))[::-1], 
                                                            pd.DataFrame((np.flip(self.sim_claims,axis=1).diagonal(axis1=1,axis2=2) - np.append(np.array([[0]]*self.n_sims),np.flip(self.sim_claims[:,:-1,:-1],axis=1).diagonal(axis1=1,axis2=2),axis=1))[:,::-1], columns = self.tri.data.index)],
                                                       'marker':['circle',None],
                                                       'color':['red',None]
                                                       }}
                    }
        return plot_dict
