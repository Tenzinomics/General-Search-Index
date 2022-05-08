
#Install google trend API first If you don't have it installed already
#!pip install pytrends #<-- this one

#Install this for the purposes of datavisualization
#!pip install matplotlib

import pandas as pd

from pytrends.request import TrendReq
pytrend = TrendReq()

from sklearn.decomposition import PCA
import statistics as stat
import numpy as np
import math

import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------------------------------------------------------------#

#The user call this function
def search_index_tinator(input_search, input_location, input_models): 

    df_search_main, search_queries  = search_index(input_search,input_location).df_search_index_tiator()
    df_search_index, df_explain_power = the_models(df_search_main,input_models).model_select_tinator()

    date = df_search_main.index

    df_search_index = pd.DataFrame({'Search Index': df_search_index})
    df_search_index = df_search_index.set_index(pd.Index(date[1:]))

    dic_search_index = {
        'data': df_search_main,
        'query': search_queries,
        'index':df_search_index
    }

    return dic_search_index


# #saerch_index class gets the monthly search data from google trend using the google tred API
class search_index:

    def __init__(self, word_search, location):
        self.word_search = word_search
        self.location = location


    def df_search_index_tiator(self):
        
        word_search = self.word_search
        geo_code_province = self.location

        #Creating an empty data frame to add search data
        df_search_main = pd.DataFrame()    
        row_length, col_length = df_search_main.shape 

        for geo_code_index in geo_code_province:
            
            #If the user have list of their own word search 
            #otherwise it uses the google search recomendation
            if isinstance(word_search, list) == True:
                search_queries = word_search

            else:
                #The initial word search
                pytrend.build_payload(kw_list=[word_search], timeframe='all', geo=geo_code_index)
                search_queries = pytrend.related_queries()[word_search]['top']['query'] #this gets the related serch words

                #the data for the initial search
                over_time_search = pytrend.interest_over_time()

                #Inserting data into dataframe
                df_search_main.insert(col_length, (geo_code_index +'_'+ word_search), over_time_search[word_search], True)
                row_length, col_length = df_search_main.shape
            

            #Loops through the search words
            for rel_query in search_queries:

                pytrend.build_payload(kw_list=[rel_query], timeframe='all', geo=geo_code_index)
                rel_query_over_time_search = pytrend.interest_over_time()

                #Inserting data into dataframe 
                df_search_main.insert(col_length, (geo_code_index +'_'+ rel_query), rel_query_over_time_search[rel_query], True)
                row_length, col_length = df_search_main.shape

        return df_search_main[180:], search_queries #by setting it on 140 it limits how far the data could go back

class the_models:

    def __init__(self, df_search, model):
        self.df_search = df_search
        self.model = model

        #df_search gets transformed in the transformation function. for now use this data frame for the average
        self.df_search_2 = df_search

    #This function is still being experimented
    def var_tranform_tinator(self):
        
        #demeaning the trend data
        for values in self.df_search.columns.values:
            u = stat.mean(self.df_search[values])
            std = stat.stdev(self.df_search[values])
            
            #Demeaning
            self.df_search[values] = (self.df_search[values] - u)/std
            
            #****Experimenting with different series transformation method******
            # self.df_search[values] = self.df_search[values].pct_change()

            # stf_pct = stat.stdev(self.df_search[values])

            # #Replacing if there are any standarddeviation
            # self.df_search[values][np.isneginf(self.df_search[values])] = stf_pct*(-1)
            # self.df_search[values][np.isinf(self.df_search[values])] = stf_pct


            # #Replacing if there are any standarddeviation
            # df_transformed[np.isneginf(df_transformed)] = std*(-1)
            # df_transformed[np.isinf(df_transformed)] = std

            #************End*******#
        

        #Experimenting with data transformation
        df_transformed = self.df_search.pct_change().dropna()
                
        return df_transformed #self.df_search

    #This function calls the function of the model the user chooses
    def model_select_tinator(self):
        dic_model = {
            1: self.pca_tinator(),

            #Expansion to the search index
            #2: 'Weighted Average'
    
        }
        return dic_model[self.model]

    ####The models#####
    def pca_tinator(self):
        
        #The data here is the transformed variable    
        df_search = self.var_tranform_tinator()

        #Principal component analysis
        pca = PCA()
        pca.fit_transform(df_search.T)
        eigen_values = pca.explained_variance_ratio_
        eigen_vectors = pca.components_

        #getting the first eigen value and eigen vectors as the search index
        df_search_index = eigen_vectors[0][:]
        df_expla_power = eigen_values[0]
        
        return df_search_index, df_expla_power


    #Future Extension. Currently work in progress

    # def avg_tinator(self):
        
    #     # arr_avg = []
    #     # for x in self.df_search_2.T:
    #     #     arr_avg.append(stat.mean(self.df_search_2.T[x]))

    #     # df_search_index_avg = pd.DataFrame()    
    #     # df_search_index_avg.insert(0, 'Average', arr_avg, True)

    #     return test #df_search_index_avg[181:], '' #remove the 181 for the full dataset
