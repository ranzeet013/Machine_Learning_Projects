#!/usr/bin/env python
# coding: utf-8

# # Predicting Video Game Sales

# The goal of this project is to build a regression model that can predict the global sales of video games based on various features such as rank, name, platform, year, genre, publisher, and regional sales data. By utilizing this model, we aim to gain insights into the factors that contribute to a game's success and help publishers and developers make informed decisions to optimize their game's performance in the market.

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

# We will start by collecting a comprehensive dataset containing information about a wide variety of video games. This dataset should include attributes such as rank, name, platform, year of release, genre, publisher, and sales figures for different regions (North America, Europe, Japan, and other regions).

# In[3]:


dataframe = pd.read_csv('video_games_sales.csv')


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


# In[9]:


dataframe.select_dtypes(include = ['int64', 'float64']).head()


# In[10]:


dataframe.isna().sum().any()


# In[11]:


dataframe.isna().sum()


# In[12]:


dataframe = dataframe.dropna()


# In[13]:


dataframe.isna().sum()


# In[14]:


dataframe.columns


# # Statical Info

# Statistical information refers to numerical data or metrics that describe various aspects of a dataset or population. These statistics provide quantitative measures of central tendency, dispersion, relationships, and other properties of the data.

# In[15]:


dataframe.describe()


# # Correlation Matrix

# A correlation matrix is a table that shows the pairwise correlations between variables in a dataset. Each cell in the table represents the correlation between two variables, and the strength and direction of the correlation is indicated by the color and magnitude of the cell.
# 
# Correlation matrices are commonly used in data analysis to identify relationships between variables and to help understand the structure of the data. The values in the correlation matrix range from -1 to 1, with -1 indicating a perfect negative correlation, 1 indicating a perfect positive correlation, and 0 indicating no correlation.

# In[16]:


corr_matrix = dataframe.corr()


# In[17]:


corr_matrix


# In[18]:


plt.figure(figsize = (12, 6))
sns.heatmap(corr_matrix, 
            annot = True, 
            cmap = 'inferno')


# In[21]:


dataset = dataframe.drop('global_sales', axis = 1)


# In[22]:


dataset.head()


# In[23]:


dataset.corrwith(dataframe['global_sales']).plot.bar(
    title = 'Correlation with Global Sales', 
    figsize = (12, 5), 
    cmap = 'plasma'
)


# In[24]:


dataframe.columns


# # Splitting Dataset

# Splitting a dataset refers to the process of dividing a given dataset into two or more subsets for training and evaluation purposes. The most common type of split is between the training set and the testing (or validation) set. This division allows us to assess the performance of a machine learning model on unseen data and evaluate its generalization capabilities.
# 
# Train-Test Split: This is the most basic type of split, where the dataset is divided into a training set and a testing set. The training set is used to train the machine learning model, while the testing set is used to evaluate its performance. The split is typically done using a fixed ratio, such as 80% for training and 20% for testing.

# In[27]:


x = dataframe[["rank", "na_sales", "eu_sales", "jp_sales", "other_sales"]]
y = dataframe["global_sales"]


# In[28]:


x.shape, y.shape


# In[29]:


from sklearn.model_selection import train_test_split


# In[30]:


x_train, x_test, y_train, y_test = train_test_split(x, 
                                                    y, 
                                                    test_size = 0.2, 
                                                    random_state = 42)


# In[31]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# # Model Selection and Training:

# For this regression task, we will explore various regression algorithms such as linear regression, decision trees, random forests, or gradient boosting to determine the best-performing model. We will split the dataset into training and testing sets to evaluate the model's accuracy and generalization capability.

# In[32]:


from sklearn.linear_model import LinearRegression


# In[33]:


model = LinearRegression()


# In[34]:


model.fit(x_train, y_train)


# In[35]:


prediction = model.predict(x_test)


# In[36]:


prediction


# In[38]:


print(y_test.iloc[20]), print(prediction[20])


# In[39]:


print(y_test.iloc[40]), print(prediction[40])


# In[42]:


model.score(x_test, y_test)


# # Thanks !
