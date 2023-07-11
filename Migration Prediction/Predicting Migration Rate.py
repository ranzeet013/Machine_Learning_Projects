#!/usr/bin/env python
# coding: utf-8

# # Predicting Migration Rate

#  The objective of this project is to develop a machine learning model that can accurately predict migration rates based on various socio-economic and demographic factors.

# # Importing Libraries

# These are just a few examples of popular Python libraries. You can import any other library using the same import statement followed by the library name or alias:
# 
# NumPy: for numerical operations and array manipulation
# 
# Pandas: for data manipulation and analysis
# 
# Matplotlib: for creating visualizations
# 
# Scikit-learn: for machine learning algorithms

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# # Importing Dataset

# In[3]:


dataframe = pd.read_csv('migration_nz.csv')


# # Data Exploration

# Data exploration is an essential step in the data analysis process. It involves examining and understanding the data to gain insights, identify patterns, and make informed decisions. 

# In[4]:


dataframe.head()


# In[5]:


dataframe.tail()


# In[6]:


dataframe.shape


# In[7]:


dataframe.info()


# In[8]:


dataframe.columns


# In[9]:


dataframe['Measure'].unique()


# In[10]:


print(dataframe['Measure'])


# In[11]:


dataframe['Measure'].replace("Arrivals", 0, inplace = True)
dataframe['Measure'].replace("Departures", 1, inplace = True)
dataframe['Measure'].replace("Net", 2, inplace = True)


# In[12]:


dataframe['Measure'].value_counts()


# In[13]:


dataframe.columns


# In[14]:


dataframe['Country'].unique()


# In[16]:


dataframe['CountryID'] = pd.factorize(dataframe.Country)[0]
dataframe['CitID'] = pd.factorize(dataframe.Citizenship)[0]


# In[18]:


dataframe['CountryID'].unique()


# # Imputing Missing Values

# In[20]:


dataframe.isnull().sum()


# In[22]:


dataframe["Value"].fillna(dataframe["Value"].median(),inplace=True)


# In[23]:


dataframe.isna().sum().any()


# In[24]:


dataframe.drop('Country', axis=1, inplace=True)
dataframe.drop('Citizenship', axis=1, inplace=True)


# # Splitting Dataset

# Splitting a dataset refers to the process of dividing a given dataset into two or more subsets for training and evaluation purposes. The most common type of split is between the training set and the testing (or validation) set. This division allows us to assess the performance of a machine learning model on unseen data and evaluate its generalization capabilities.
# 
# Train-Test Split: This is the most basic type of split, where the dataset is divided into a training set and a testing set. The training set is used to train the machine learning model, while the testing set is used to evaluate its performance. The split is typically done using a fixed ratio, such as 70% for training and 30% for testing.

# In[25]:


from sklearn.model_selection import train_test_split


# In[28]:


X = dataframe[['CountryID', 'Measure', 'Year', 'CitID']].values
Y = dataframe['Value'].values


# In[31]:


X.shape, Y.shape


# In[32]:


X_train, X_test, Y_train, Y_test = train_test_split(X, 
                                                    Y, 
                                                    test_size = 0.3, 
                                                    random_state = 42)


# In[33]:


X_train.shape, X_test.shape, Y_train.shape, Y_test.shape


# # Random Forest Regressor

# Random Forest Regressor is a machine learning algorithm that is based on the Random Forest ensemble method and used for regression tasks. It is an extension of the Random Forest Classifier, but instead of predicting categorical variables, it predicts continuous numeric values.

# In[35]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=70,max_features = 3,max_depth=5,n_jobs=-1)
model.fit(X_train ,Y_train)
model.score(X_test, Y_test)


# # Migration Rate

# In[39]:


X = dataframe[['CountryID','Measure','Year','CitID']]
Y = dataframe['Value']
X_train, X_test, y_train, y_test = train_test_split(
  X, Y, test_size=0.3, random_state=9)
grouped = dataframe.groupby(['Year']).aggregate({'Value' : 'sum'})


grouped.plot(kind='line');plt.axhline(0, color='g')
plt.show()


# In[41]:


corr = dataframe.corr()


# In[42]:


corr


# In[44]:


sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.show()

