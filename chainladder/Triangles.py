# -*- coding: utf-8 -*-


 
class triangle:
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
          
    def dataAsTable(self, inplace=False):
        # will need to create triangle class that has origin and dev
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
            return
        

    def dataAsTriangle(self, inplace=False):
        if self.dataform == 'tabular':
            tri = pivot_table(self.data,values=self.values,index=[self.origin], columns=[self.dev]).sort_index()
            tri.columns = [str(item) for item in tri.columns]
            if inplace == True:
                self.data = tri   
                self.dataform = 'triangle'
        return tri
        
    def incr2cum(self, inplace=False):
        incr = DataFrame(self.data.iloc[:,0])
        for val in range(1, len(self.data.T.index)):
            incr = concat([incr,self.data.iloc[:,val]+incr.iloc[:,-1]],axis=1)
        incr = incr.rename_axis('dev', axis='columns')
        incr.columns = self.data.T.index
        if inplace == True:
            self.data = incr
        return incr     
    
    def cum2incr(self, inplace=False):
        incr = self.data.iloc[:,0]
        for val in range(1, len(self.data.T.index)):
            incr = concat([incr,self.data.iloc[:,val]-self.data.iloc[:,val-1]],axis=1)
        incr = incr.rename_axis('dev', axis='columns')
        incr.columns = self.data.T.index
        if inplace == True:
            self.data = incr        
        return incr   
    
    def __set_origin(self):
        ##### Identify Origin Profile ####
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
        ##### Identify dev Profile ####
        value_names = ['incurred claims'] 
        values_match = [i for i in value_names if i in self.data.columns]
        if len(values_match)==1:
            self.values = values_match[0]
        else:
            self.values = 'values'
        return 



def plot(tri):
    # Need lattice (cascade) option
    plt.figure(figsize=(25,15))
    for i in range(len(tri.data)):
        plt.plot(tri.data.columns.astype(int),tri.data.iloc[i])
        
    
    
    
    
    