"""
This module contains various utilities shared across most of the other
*chainladder* modules.

"""
from pandas import read_pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot


def load_dataset(key):
    """ Function to load datasets included in the chainladder package.
   
    Arguments:
	key: str
	    The name of the dataset, e.g. RAA, ABC, UKMotor, GenIns, etc.
    
    Returns:
	pandas.DataFrame of the loaded dataset.
   """
    path = os.path.dirname(os.path.abspath(__file__))
    return read_pickle(os.path.join(path, 'data', key))

    
class Plot():
    """ Class that renders the matplotlib plots 
    based on the configurations in __get_plot_dict().

    Arguments:
        my_dict: dict
            A dictionary representing the charts the end user would like
            to see.  If ommitted, the default plots are displayed.

    Returns:
        Renders the matplotlib plots.

    """       
    def __init__(self, ctype, my_dict, **kwargs): 
        plot_list = []
        self.grid = None
        if ctype=='m':
            sns.set_style("whitegrid")
            _ = plt.figure()
            grid_x = 1 if len(my_dict) == 1 else round(len(my_dict) / 2,0)
            grid_y = 1 if len(my_dict) == 1 else 2
            fig, ax = plt.subplots(figsize=(grid_y*15, grid_x*10))
            for num, item in enumerate(my_dict):
                _ = plt.subplot(grid_x,grid_y,num+1)
                self.__dict_plot(item) 
        if ctype[0]=='b':
            for num, item in enumerate(my_dict):
                plot_list.append(self.__bokeh_dict_plot(item))
            grid = gridplot(plot_list,  ncols=2, **kwargs)
            if len(ctype) == 1:
                show(grid)
            self.grid = grid

    def __dict_plot(self, my_dict):
        """ Method that renders the matplotlib plots based on the configurations
        in __get_plot_dict().  This method should probably be private and may be
        so in a future release.
        
        """
        for i in range(len(my_dict['chart_type_dict']['mtype'])):
            if my_dict['chart_type_dict']['mtype'][i] == 'plot':
                _ = plt.plot(my_dict['chart_type_dict']['x'][i], 
                             my_dict['chart_type_dict']['yM'][i],
                             linestyle=my_dict['chart_type_dict']['linestyle'][i], 
                             marker=my_dict['chart_type_dict']['markerM'][i], 
                             color=my_dict['chart_type_dict']['colorM'][i])
            if my_dict['chart_type_dict']['mtype'][i] == 'bar':
                _ =plt.bar(height = my_dict['chart_type_dict']['height'][i], 
                           left = my_dict['chart_type_dict']['x'][i],
                           yerr = my_dict['chart_type_dict']['yerr'][i], 
                           bottom = my_dict['chart_type_dict']['bottom'][i],
                           width = my_dict['chart_type_dict']['width'][i])
            if my_dict['chart_type_dict']['mtype'][i] == 'line':
                _ = plt.gca().set_prop_cycle(None)
                for j in range(my_dict['chart_type_dict']['rows'][i]):
                    _ = plt.plot(my_dict['chart_type_dict']['x'][i], 
                                 my_dict['chart_type_dict']['y'][i].iloc[j], 
                                 linestyle=my_dict['chart_type_dict']['linestyle'][i], 
                                 linewidth=my_dict['chart_type_dict']['line_width'][i],
                                 alpha=my_dict['chart_type_dict']['alpha'][i])
            if my_dict['chart_type_dict']['mtype'][i] == 'hist':
                _ = sns.distplot(my_dict['chart_type_dict']['x'][i],
                                 kde=True,)
            if my_dict['chart_type_dict']['mtype'][i] == 'box':
                _ = plt.boxplot(my_dict['chart_type_dict']['x'][i],
                                positions= my_dict['chart_type_dict']['positions'][i],
                                showfliers=True)
            _ = plt.title(my_dict['Title'], fontsize=30)
            _ = plt.xlabel(my_dict['XLabel'], fontsize=20)
            _ = plt.ylabel(my_dict['YLabel'],fontsize=20)
    
    def __bokeh_dict_plot(self, my_dict):
        """ Method that renders the Bokeh plots based on the configurations
        in __get_plot_dict().
        
        """
        p = figure(plot_width=450, plot_height=275,title=my_dict['Title'], x_axis_label = my_dict['XLabel'], y_axis_label=my_dict['YLabel'], logo=None)
        for i in range(len(my_dict['chart_type_dict']['type'])):
            if my_dict['chart_type_dict']['type'][i] == 'scatter':
                p.scatter(x = my_dict['chart_type_dict']['x'][i], 
                          y = my_dict['chart_type_dict']['y'][i], 
                          legend = my_dict['chart_type_dict']['label'][i],
                          color = my_dict['chart_type_dict']['color'][i],
                          alpha = my_dict['chart_type_dict']['alpha'][i],
                          marker = my_dict['chart_type_dict']['marker'][i])         
            if my_dict['chart_type_dict']['type'][i] == 'vbar':
                p.vbar(x = my_dict['chart_type_dict']['x'][i], 
                      top = my_dict['chart_type_dict']['top'][i],
                      color = my_dict['chart_type_dict']['color'][i], 
                      bottom = my_dict['chart_type_dict']['bottom'][i],
                      width = my_dict['chart_type_dict']['width'][i],
                      legend = my_dict['chart_type_dict']['label'][i])
            if my_dict['chart_type_dict']['type'][i] == 'line':
                for j in range(my_dict['chart_type_dict']['rows'][i]):
                    p.line(x = my_dict['chart_type_dict']['x'][i], 
                           y = my_dict['chart_type_dict']['y'][i].iloc[j], 
                           line_width=my_dict['chart_type_dict']['line_width'][i], 
                           color=my_dict['chart_type_dict']['color'][i][j], 
                           line_cap=my_dict['chart_type_dict']['line_cap'][i], 
                           line_join=my_dict['chart_type_dict']['line_join'][i],
                           line_dash=my_dict['chart_type_dict']['line_dash'][i],
                           legend=my_dict['chart_type_dict']['label'][i][j],
                           alpha=my_dict['chart_type_dict']['alpha'][i])
            if my_dict['chart_type_dict']['type'][i] == 'errline':                
                    p.multi_line(xs = my_dict['chart_type_dict']['x'][i], 
                           ys = my_dict['chart_type_dict']['y'][i],
                           color=my_dict['chart_type_dict']['color'][i])                                                     
            if my_dict['chart_type_dict']['type'][i] == 'quad':
                p.quad(top=my_dict['chart_type_dict']['top'][i],
                       bottom=my_dict['chart_type_dict']['bottom'][i],
                       left=my_dict['chart_type_dict']['left'][i],
                       right=my_dict['chart_type_dict']['right'][i],
                       legend=my_dict['chart_type_dict']['label'][i],
                       alpha=0.5,line_color='black')
            if my_dict['chart_type_dict']['type'][i] == 'box':
                p = boxwhisker(p,my_dict['chart_type_dict']['y'][i])
        return p

def boxwhisker(fig, a):
    import pandas as pd
    from bokeh.plotting import figure
    
    a.columns = [str(column) for column in a.columns]
    df=pd.DataFrame()
    for num, column in enumerate(a):
        b = pd.Series(a.iloc[:,num], name='score')
        b.index = [a.iloc[:,num].name]*len(a)
        df = df.append(pd.DataFrame(b))
    cats = list(df.index.unique())
    groups = df.groupby(df.index)
    q1 = groups.quantile(q=0.25)
    q2 = groups.quantile(q=0.5)
    q3 = groups.quantile(q=0.75)
    iqr = q3 - q1
    upper = q3 + 1.5*iqr
    lower = q1 - 1.5*iqr
    
    # find the outliers for each category
    def outliers(group):
        cat = group.name
        return group[(group.score > upper.loc[cat]['score']) | (group.score < lower.loc[cat]['score'])]['score']
    out = groups.apply(outliers).dropna()
      
    # prepare outlier data for plotting, we need coordinates for every outlier.
    if not out.empty:
        outx = []
        outy = []
        for cat in cats:
            # only add outliers if they exist
            if not out.loc[cat].empty:
                for value in out.loc[cat]:
                    outx.append(cat)
                    outy.append(value)
    
    p = fig   
    # if no outliers, shrink lengths of stems to be no longer than the minimums or maximums
    qmin = groups.quantile(q=0.00)
    qmax = groups.quantile(q=1.00)
    upper.score = [min([x,y]) for (x,y) in zip(list(qmax.loc[:,'score']),upper.score)]
    lower.score = [max([x,y]) for (x,y) in zip(list(qmin.loc[:,'score']),lower.score)]
    
    # stems
    p.segment(cats, upper.score, cats, q3.score, line_color="black")
    p.segment(cats, lower.score, cats, q1.score, line_color="black")
    
    # boxes
    p.vbar(cats, 0.7, q2.score, q3.score, fill_color="lightblue", fill_alpha = 0.5, line_color="black")
    p.vbar(cats, 0.7, q1.score, q2.score, fill_color="lightblue", fill_alpha = 0.5, line_color="black")
    
    # whiskers (almost-0 height rects simpler than segments)
    p.rect(cats, lower.score, 0.2, 0.01, line_color="black")
    p.rect(cats, upper.score, 0.2, 0.01, line_color="black")
    
    # outliers
    if not out.empty:
        p.circle(outx, outy, color="grey", fill_alpha=0.6)
    return p
            
            


            