#!/usr/bin/env python
# coding: utf-8

# # Mobile Price Classification

# The goal of this project is to develop a machine learning model that can classify mobile phone prices into different categories based on their features. By using this model, users will be able to predict the price range of a mobile phone based on its specifications, helping them make informed decisions while purchasing a new phone.

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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# # Importing Dataset

# In[3]:


dataframe = pd.read_csv('mobileprice.csv')


# # Data Processing

# Data preprocessing plays a crucial role in the success of any machine learning project, including mobile price classification.

# In[4]:


dataframe.head()


# In[5]:


dataframe.tail()


# In[6]:


dataframe.shape


# In[7]:


dataframe.info()


# In[8]:


dataframe.isna().any()


# In[9]:


dataframe.columns


# # Statical Info

# In addition to data processing, it is important to gather statistical information about the dataset to gain insights and understand the characteristics of the mobile phone prices. 

# In[10]:


dataframe.describe()


# # Correlation Matrix

# 
# A correlation matrix is a table that displays the correlation coefficients between multiple variables in a dataset. In the context of the mobile price classification project, a correlation matrix can provide insights into the relationships between different mobile phone features and their impact on prices.

# In[11]:


corr_matrix = dataframe.corr()


# In[12]:


corr_matrix


# In[15]:


plt.figure(figsize = (18, 10))
sns.heatmap(corr_matrix, 
            annot = True, 
            cmap = 'inferno')


# In[16]:


dataframe.columns


# # Correlation Of Price Range 

# In[17]:


dataset = dataframe.drop('price_range', axis = 1)


# In[18]:


dataset.head()


# In[20]:


dataset.corrwith(dataframe['price_range']).plot.bar(
    figsize = (12, 5),
    title = 'Correlation Of Price', 
    cmap = 'plasma'
)


# In[21]:


dataframe.head()


# # Scaling

# Scaling is an important preprocessing step in machine learning that aims to normalize the range of numerical features in the dataset. In the context of the mobile price classification project, scaling ensures that all the features contribute equally to the model training process.

# In[23]:


x = dataframe.iloc[:, :-1].values
y = dataframe.iloc[:, -1].values


# In[24]:


x.shape, y.shape


# In[25]:


from sklearn.preprocessing import StandardScaler


# In[26]:


scaler = StandardScaler()


# In[27]:


x = scaler.fit_transform(x)


# # Splitting Dataset

# 
# Splitting the dataset is an essential step in machine learning to evaluate and validate the performance of the model. In the context of the mobile price classification project, you need to divide the dataset into training and testing sets.

# In[28]:


from sklearn.model_selection import train_test_split


# In[29]:


x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y, 
                                                    test_size = 0.2, 
                                                    random_state = 41)


# In[30]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[31]:


from sklearn.linear_model import LogisticRegression


# In[32]:


log = LogisticRegression()


# In[33]:


log.fit(x_train, y_train)


# # Prediction On x-Test 

# In[34]:


y_pred = log.predict(x_test)


# In[35]:


y_pred


# In[36]:


print(y_test[20]), print(y_pred[20])


# # Error Analysis

# Error analysis is a crucial step in machine learning projects, including the mobile price classification project, to gain insights into the model's performance, identify sources of errors, and determine areas for improvement.

# In[37]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[38]:


accuracy_score = accuracy_score(y_test, y_pred)


# In[39]:


accuracy_score


# In[40]:


confusion_matrix = confusion_matrix(y_test, y_pred)


# In[41]:


confusion_matrix


# In[42]:


plt.figure(figsize = (6, 4))
sns.heatmap(confusion_matrix, 
            annot = True, 
            cmap = 'RdPu')


# In[43]:


print(classification_report(y_test, y_pred))


# # Thanks !
