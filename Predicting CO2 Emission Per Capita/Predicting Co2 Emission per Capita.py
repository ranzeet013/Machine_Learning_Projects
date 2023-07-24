#!/usr/bin/env python
# coding: utf-8

# # Predicting Co2 Emission per Capita

# The CO2 Emission Per Capita Prediction project aims to develop a data-driven solution that accurately forecasts carbon dioxide (CO2) emissions on a per capita basis for different regions or countries. The project seeks to leverage historical emissions data, socio-economic indicators, and potentially other relevant variables to create a predictive model capable of estimating future emissions per person. The ultimate goal is to gain insights into emission trends, identify factors influencing emissions, and aid policymakers in making informed decisions to mitigate climate change.

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
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# # Importing Dataset

# The dataset contains monthly observations of environmental, atmospheric, and socio-economic attributes, including greenhouse gases like CO2, CH4, and N2O, as well as MEI for El Niño analysis. It also includes CFC-11 and CFC-12 data, TSI for solar irradiance, aerosols, temperature, population, and CO2 emissions per capita.

# In[3]:


dataframe = pd.read_csv('carbon-segment.csv')


# # Data Processing 

# After acquiring the dataset, we will perform data preprocessing to clean and prepare the data for analysis. This includes handling missing values, removing duplicates, and transforming categorical variables into numerical representations using techniques like one-hot encoding or label encoding.

# In[4]:


dataframe.head()


# In[5]:


dataframe.tail()


# In[6]:


dataframe.shape


# In[7]:


dataframe.info()


# In[8]:


dataframe.isna().sum().any()


# In[9]:


dataframe.isna().any()


# In[10]:


dataframe.columns


# # Visualizing Co2 Emission in Different Interval Of Time

# From the diagram below we can see that the emission of Co2 has increased rapidly in last 15 years.

# In[16]:


plt.rcParams['figure.figsize'] = [12, 5]

ax = sns.lineplot(data = dataframe, x = 'Year', y = 'CO2emissionspercapita')

from matplotlib.ticker import PercentFormatter
ax.yaxis.set_major_formatter(PercentFormatter(1.0))


# # Statical Info

# Statistical information refers to numerical data or metrics that describe various aspects of a dataset or population. These statistics provide quantitative measures of central tendency, dispersion, relationships, and other properties of the data.

# In[17]:


dataframe.describe()


# # Correlation Matrix

# A correlation matrix is a table that shows the pairwise correlations between variables in a dataset. Each cell in the table represents the correlation between two variables, and the strength and direction of the correlation is indicated by the color and magnitude of the cell.
# 
# Correlation matrices are commonly used in data analysis to identify relationships between variables and to help understand the structure of the data. The values in the correlation matrix range from -1 to 1, with -1 indicating a perfect negative correlation, 1 indicating a perfect positive correlation, and 0 indicating no correlation.

# In[18]:


corr_matrix = dataframe.corr()


# In[19]:


corr_matrix


# In[20]:


plt.figure(figsize = (12, 6))
sns.heatmap(corr_matrix, 
            annot = True, 
            cmap = 'inferno')


# # Correlation Of Co2 Emission With Different Attributes

# In[22]:


dataframe.columns


# In[24]:


dataset = dataframe.drop('CO2emissionspercapita', axis = 1)


# In[25]:


dataset.head()


# In[26]:


dataset.corrwith(dataframe['CO2emissionspercapita']).plot.bar(
    title = 'Correlation With Co2 Emission Per Capita', 
    figsize = (12, 5), 
    cmap = 'plasma'
)


# In[27]:


dataframe.head()


# # Splitting Dataset

# Splitting a dataset refers to the process of dividing a given dataset into two or more subsets for training and evaluation purposes. The most common type of split is between the training set and the testing (or validation) set. This division allows us to assess the performance of a machine learning model on unseen data and evaluate its generalization capabilities.
# 
# Train-Test Split: This is the most basic type of split, where the dataset is divided into a training set and a testing set. The training set is used to train the machine learning model, while the testing set is used to evaluate its performance. The split is typically done using a fixed ratio, such as 80% for training and 20% for testing.

# In[28]:


x = dataframe.drop('CO2emissionspercapita', axis = 1)
y = dataframe['CO2emissionspercapita']


# In[29]:


x.shape, y.shape


# In[30]:


from sklearn.model_selection import train_test_split


# In[32]:


x_train, x_test, y_train, y_test = train_test_split(x, 
                                                    y, 
                                                    test_size = 0.2, 
                                                    random_state = 42)


# In[33]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# # Model Selection And Training

# In[37]:


from sklearn.linear_model import LinearRegression


# In[38]:


model = LinearRegression()


# In[39]:


model.fit(x_train, y_train)


# In[40]:


y_pred = model.predict(x_test)


# In[41]:


y_pred


# In[42]:


model.score(x_test, y_test)


# In[43]:


from sklearn.ensemble import RandomForestRegressor


# In[44]:


model_rfg = RandomForestRegressor()


# In[45]:


model_rfg.fit(x_train, y_train)


# In[46]:


y_pred_rfg = model_rfg.predict(x_test)


# In[47]:


y_pred_rfg


# In[48]:


model_rfg.score(x_test, y_test)


# # Thanks !
