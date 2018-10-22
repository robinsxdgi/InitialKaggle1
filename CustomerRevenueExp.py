
# coding: utf-8

# # This program is used for exploring the data structures of Google customer revenue training data file
# 
# ## 1. Import the necessary library



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
from pandas.io.json import json_normalize #package for flattening json in pandas df
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../GoogleCustomerRevenue"))



# Any results you write to the current directory are saved as output.

# Let's explore the data a little bit
traindf = pd.read_csv("../GoogleCustomerRevenue/train.csv")
print(traindf.head(5))


# ### By observation, we can find that there are three columns that are with Json type. "device", "geoNetwork", "totals", "trafficSource" need to be converted to dictrionary type. 

# # 2. Define a function that converts a file with Json columns to a normal table file.



def json_to_df (file_path = "../GoogleCustomerRevenue/train.csv", json_columns = ['device', 'geoNetwork', 'totals', 'trafficSource']):
    # json_columns is a list consists of names of columns that are written in Json.
    
    df = pd.read_csv(file_path, 
                     converters={column: json.loads for column in json_columns}, 
                     # Since json_normalize()'s inputs are dict or list of dicts, the string-json columns need to be converted
                     # for using converters, see https://stackoverflow.com/questions/20680272/parsing-a-json-string-which-was-loaded-from-a-csv-using-pandas
                     dtype={'fullVisitorId':'str'}# to make sure ID is unique)
       
    for column in json_columns:
        flatten_json = json_normalize(df[column])
        # Usage of json_normalize() can be found in 
        #   https://pandas.pydata.org/pandas-docs/stable/generated/pandas.io.json.json_normalize.html
        # When the json file has embedded arrays, see
        #   https://www.kaggle.com/jboysen/quick-tutorial-flatten-nested-json-in-pandas
        #   https://towardsdatascience.com/flattening-json-objects-in-python-f5343c794b10
        
        flatten_json = flatten_json.rename(columns = lambda x: column + '_' + x, inplace = True)
        # rename the columns by the format of "column_labels"
        
        df = df.drop(column, axis = 1).merge(flatten_json, left_index = True, right_index = True)
        #for more on merge, see https://www.shanelynn.ie/merge-join-dataframes-python-pandas-index-1/#differentnames
        
    return df


# ### Convert the training data and test data to the cleaned data frame by the above function


traindf = json_to_df()


testdf = json_to_df("../GoogleCustomerRevenue/test.csv")


# ### Let's see if the new columns are all flattend


pd.options.display.max_columns = None
traindf.head()


# ### Wow, UC Browser!!!


testdf.head()


# ### Cool, now let's save these two flattened files


get_ipython().run_cell_magic('time', '', 'traindf.to_csv("../GoogleCustomerRevenue/flattened_train.csv", index=False)#index=false won\'t add number index before rows\ntestdf.to_csv("../GoogleCustomerRevenue/flattened_test.csv", index=False)')


flat_traindf = pd.read_csv("../GoogleCustomerRevenue/flattened_train.csv", 
                           dtype = {'totals_transactionRevenue' : float,
                                   'totals_bounces': float,
                                   'totals_hits': float,
                                   'totals_newVisits' : float,
                                   'totals_pageviews': float,
                                   'trafficSource_adwordsClickInfo.page': float})
flat_testdf = pd.read_csv("../GoogleCustomerRevenue/flattened_test.csv",
                          dtype = {'totals_bounces': float,
                                   'totals_hits': float,
                                   'totals_newVisits' : float,
                                   'totals_pageviews': float,
                                   'trafficSource_adwordsClickInfo.page': float})


len(set(testdf['fullVisitorId']))



#Thanks to Rahul, we are able to quickly see the statistic summary of the dataframe by using the following function

#  https://www.kaggle.com/rahullalu/gstore-eda-lgbm-baseline-1-4281

#FUNCTION FOR PROVIDING FEATURE SUMMARY
def feature_summary(df_fa):
    print('DataFrame shape')
    print('rows:',df_fa.shape[0])
    print('cols:',df_fa.shape[1])
    
    col_list=['Null','Unique_Count','Data_type','Max/Min','Mean','Std','Skewness','Sample_values']#some basic features
    df=pd.DataFrame(index=df_fa.columns,columns=col_list)
    
    df['Null']=list([len(df_fa[col][df_fa[col].isnull()]) for i,col in enumerate(df_fa.columns)])
    df['Unique_Count']=list([len(df_fa[col].unique()) for i,col in enumerate(df_fa.columns)])
    df['Data_type']=list([df_fa[col].dtype for i,col in enumerate(df_fa.columns)])
    for i,col in enumerate(df_fa.columns):
        
        #When the data type is numbers, we can calculate their statistic properties
        if 'float' in str(df_fa[col].dtype) or 'int' in str(df_fa[col].dtype):
            df.at[col,'Max/Min']=str(round(df_fa[col].max(),2))+'/'+str(round(df_fa[col].min(),2))
            df.at[col,'Mean']=df_fa[col].mean()
            df.at[col,'Std']=df_fa[col].std()
            df.at[col,'Skewness']=df_fa[col].skew()
        
        #Show some samples of each column
        df.at[col,'Sample_values']=list(df_fa[col].unique())
           
    return(df.fillna('-'))


feature_summary(flat_traindf)


feature_summary(flat_testdf)


#notice that the data in transactionRevenue has a lot of NA
#so we want to replace nan to 0

flat_traindf.totals_transactionRevenue = flat_traindf.totals_transactionRevenue.fillna(0)


#ADDING ANOTHER FEATURE revenue_status TO INDICATE PRESENCE/ABSENCE OF REVENUE FOR EACH OBSERVATION
flat_traindf['revenue_status'] = flat_traindf.totals_transactionRevenue.apply(lambda x: 0 if x==0 else 1)


flat_traindf.head()


# # 3. Start to Explore Data Propeties

# ## The percentage of transactions that generates revenue


revenue_num = flat_traindf.revenue_status.value_counts()[1] #number of transactions generate revene
percent_revenue = revenue_num / len(flat_traindf.revenue_status)

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Transactions with revene', 'Transactions without revene'
sizes = [percent_revenue, 1 - percent_revenue]
explode = (0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots(figsize = (5,5))
pie_colors=('white','peachpuff')
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=45, colors = pie_colors)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# ## Now we take a look at the revenue generated by different browsers


browser_rank = flat_traindf[['device_browser',
                             'totals_transactionRevenue',
                             'revenue_status']].groupby('device_browser').sum().reset_index()
browser_rank = browser_rank.sort_values(by = 'totals_transactionRevenue', 
                                        ascending = False)[browser_rank.totals_transactionRevenue > 0]
                                        #only focus on the browsers that generate positive revenue
plt.figure(figsize=(12,6))
plt.bar(range(browser_rank.shape[0]), np.log1p(browser_rank.totals_transactionRevenue), color = 'darkgray')
plt.xticks(range(browser_rank.shape[0]),browser_rank.device_browser,rotation=45,fontsize=12)
plt.xlabel('Browsers',fontsize=18)
plt.ylabel('Log Revenue',fontsize=18)
plt.show()



browser_rank = flat_traindf[['device_browser',
                             'totals_transactionRevenue',
                             'revenue_status']].groupby('device_browser').sum().reset_index()
browser_rank = browser_rank.sort_values(by = 'revenue_status', 
                                        ascending = False)[browser_rank.revenue_status > 0]
                                        #only focus on the browsers that generate positive revenue
plt.figure(figsize=(12,6))
plt.bar(range(browser_rank.shape[0]), np.log1p(browser_rank.revenue_status), color = 'dodgerblue')
plt.xticks(range(browser_rank.shape[0]),browser_rank.device_browser,rotation=45,fontsize=12)
plt.xlabel('Browsers',fontsize=18)
plt.ylabel('Number of Transcations with Revenue',fontsize=18)
plt.show()

