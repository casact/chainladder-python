"""
This module contains various utilities shared across most of the other
*chainladder* modules.

"""
from pandas import read_pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns


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
    """ Method, callable by end-user that renders the matplotlib plots 
    based on the configurations in __get_plot_dict().

    Arguments:
        my_dict: dict
            A dictionary representing the charts the end user would like
            to see.  If ommitted, the default plots are displayed.

    Returns:
        Renders the matplotlib plots.

    """       
    def __init__(self, my_dict):     
        sns.set_style("whitegrid")
        _ = plt.figure()
        grid_x = 1 if len(my_dict) == 1 else round(len(my_dict) / 2,0)
        grid_y = 1 if len(my_dict) == 1 else 2
        fig, ax = plt.subplots(figsize=(grid_y*15, grid_x*10))
        for num, item in enumerate(my_dict):
            _ = plt.subplot(grid_x,grid_y,num+1)
            self.__dict_plot(item) 

    def __dict_plot(self, my_dict):
        """ Method that renders the matplotlib plots based on the configurations
        in __get_plot_dict().  This method should probably be private and may be
        so in a future release.
        
        """
        for i in range(len(my_dict['chart_type_dict']['type'])):
            if my_dict['chart_type_dict']['type'][i] == 'plot':
                _ = plt.plot(my_dict['chart_type_dict']['x'][i], 
                             my_dict['chart_type_dict']['y'][i],
                             linestyle=my_dict['chart_type_dict']['linestyle'][i], 
                             marker=my_dict['chart_type_dict']['marker'][i], 
                             color=my_dict['chart_type_dict']['color'][i])
            if my_dict['chart_type_dict']['type'][i] == 'bar':
                _ =plt.bar(height = my_dict['chart_type_dict']['height'][i], 
                           left = my_dict['chart_type_dict']['left'][i],
                           yerr = my_dict['chart_type_dict']['yerr'][i], 
                           bottom = my_dict['chart_type_dict']['bottom'][i],
                           width = my_dict['chart_type_dict']['width'][i])
            if my_dict['chart_type_dict']['type'][i] == 'line':
                _ = plt.gca().set_prop_cycle(None)
                for j in range(my_dict['chart_type_dict']['rows'][i]):
                    _ = plt.plot(my_dict['chart_type_dict']['x'][i], 
                                 my_dict['chart_type_dict']['y'][i].iloc[j], 
                                 linestyle=my_dict['chart_type_dict']['linestyle'][i], 
                                 linewidth=my_dict['chart_type_dict']['linewidth'][i],
                                 alpha=my_dict['chart_type_dict']['alpha'][i])
            if my_dict['chart_type_dict']['type'][i] == 'hist':
                _ = plt.hist(my_dict['chart_type_dict']['x'][i],
                             bins=my_dict['chart_type_dict']['bins'][i])
            _ = plt.title(my_dict['Title'], fontsize=30)
            _ = plt.xlabel(my_dict['XLabel'], fontsize=20)
            _ = plt.ylabel(my_dict['YLabel'],fontsize=20)
            
