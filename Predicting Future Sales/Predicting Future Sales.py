#!/usr/bin/env python
# coding: utf-8

# # Predicting Future Sales

# The goal of the "Future Sales Prediction" project is to develop a machine learning model that can forecast future sales for a given set of products or items. The project aims to assist businesses in making informed decisions related to inventory management, production planning, and sales forecasting.

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


# # Importing Datasets

# The project utilizes historical sales data, which includes information such as sale, product/item ID,  and corresponding sales quantity.

# In[3]:


dataframe = pd.read_csv('sales.csv')


# # Exploratory Data Analysis

# 
# Exploratory Data Analysis (EDA) is a crucial step in the "Future Sales Prediction" project as it helps to understand the dataset, identify patterns, and gain insights that can guide the modeling process

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


# # Data Visualization

# Data visualization plays a crucial role in exploratory data analysis and provides meaningful insights into the "Future Sales Prediction" project. Visualizations help to understand patterns, relationships, and distributions within the dataset

# In[13]:


figure = plt.scatter(data = dataframe , x="Sales",
                    y="TV")


# In[14]:


figure = plt.scatter(data = dataframe , x="Sales",
                    y="Newspaper")


# In[16]:


figure = plt.scatter(data = dataframe , x="Sales",
                    y="Radio")


# In[17]:


dataframe.columns


# # Statical Info

# Statistical information refers to numerical data or metrics that describe various aspects of a dataset or population. These statistics provide quantitative measures of central tendency, dispersion, relationships, and other properties of the data.

# In[18]:


dataframe.describe()


# # Correlation Matrix

# A correlation matrix is a table that shows the pairwise correlations between variables in a dataset. Each cell in the table represents the correlation between two variables, and the strength and direction of the correlation is indicated by the color and magnitude of the cell.
# 
# Correlation matrices are commonly used in data analysis to identify relationships between variables and to help understand the structure of the data. The values in the correlation matrix range from -1 to 1, with -1 indicating a perfect negative correlation, 1 indicating a perfect positive correlation, and 0 indicating no correlation.

# In[19]:


corr_matrix = dataframe.corr()


# In[20]:


corr_matrix


# In[21]:


plt.figure(figsize = (9, 7))
sns.heatmap(corr_matrix, 
            annot = True, 
            cmap = 'inferno')


# In[22]:


dataframe.head()


# In[23]:


dataset = dataframe.drop('Sales', axis = 1)


# In[24]:


dataset.head()


# In[25]:


dataset.corrwith(dataframe['Sales']).plot.bar(
    figsize = (10, 5), 
    cmap = 'plasma', 
    title = 'Correlation with TV, Radio and Newspaper'
)


# In[26]:


dataframe.head()


# # Splitting Dataset

# Splitting a dataset refers to the process of dividing a given dataset into two or more subsets for training and evaluation purposes. The most common type of split is between the training set and the testing (or validation) set. This division allows us to assess the performance of a machine learning model on unseen data and evaluate its generalization capabilities.
# 
# Train-Test Split: This is the most basic type of split, where the dataset is divided into a training set and a testing set. The training set is used to train the machine learning model, while the testing set is used to evaluate its performance. The split is typically done using a fixed ratio, such as 80% for training and 20% for testing.

# In[27]:


x = np.array(dataframe.drop(["Sales"], 1))
y = np.array(dataframe["Sales"])


# In[28]:


from sklearn.model_selection import train_test_split


# In[29]:


x_train, x_test, y_train, y_test = train_test_split(x, 
                                                    y, 
                                                    test_size = 0.2, 
                                                    random_state = 42)


# In[30]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# # Model Training

# 
# Model training is a crucial step in the "Future Sales Prediction" project, where you develop a machine learning model using historical sales data. The trained model will then be used to make predictions on future sales.

# In[31]:


from sklearn.linear_model import LinearRegression


# In[32]:


model = LinearRegression()


# In[33]:


model.fit(x_train, y_train)


# In[34]:


print(model.score(x_test, y_test))


# In[35]:


features = np.array([[230.1, 37.8, 69.2]])
print(model.predict(features))


# # Predicting on x-Test

# In[36]:


y_pred = model.predict(x_test)


# In[37]:


y_pred


# In[38]:


print(y_test[20]), print(y_pred[20])


# # Thanks !
