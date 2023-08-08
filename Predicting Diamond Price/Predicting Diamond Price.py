#!/usr/bin/env python
# coding: utf-8

# # Predicting Diamond Price 

# In this data science project, i will develop a predictive model to estimate the price of diamonds based on their characteristics. Diamonds are not only precious gemstones but also carry a wide range of attributes that influence their value, such as carat weight, cut quality, color, and clarity. By creating a predictive model, you can help diamond buyers and sellers make more informed decisions and gain insights into the factors that drive diamond prices.

# # 1.Importing LIbraries :

# Pandas: Pandas is a powerful and widely used library for data manipulation and analysis. It provides data structures like DataFrames and Series, which allow you to store and manipulate tabular data. Pandas offers a wide range of functions for data cleaning, filtering, aggregation, merging, and more. It also supports reading and writing data from various file formats such as CSV, Excel, SQL databases, and more.
# 
# NumPy: NumPy (Numerical Python) is a fundamental library for scientific computing in Python. It provides efficient data structures like arrays and matrices and a vast collection of mathematical functions. NumPy enables you to perform various numerical operations on large datasets, such as element-wise calculations, linear algebra, Fourier transforms, and random number generation. It also integrates well with other libraries for data analysis and machine learning.
# 
# Matplotlib: Matplotlib is a popular plotting library that enables you to create a wide range of static, animated, and interactive visualizations. It provides a MATLAB-like interface and supports various types of plots, including line plots, scatter plots, bar plots, histograms, and more. Matplotlib gives you extensive control over plot customization, including labels, colors, legends, and annotations, allowing you to effectively communicate insights from your data
# 
# Seaborn: Seaborn is a Python library for creating attractive and insightful statistical visualizations, particularly suited for categorical data, with built-in themes, improved aesthetics, and seamless integration with Pandas DataFrames.

# In[58]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# # 2.Importing Dataset :

# The dataset meticulously presents attributes that intricately detail the characteristics of individual diamonds. Each diamond is defined by attributes like carat weight, influencing its mass, and cut quality, impacting visual brilliance. Color grade reveals hue, clarity detects imperfections, and depth provides proportion insight. Table width influences aesthetics. Price quantifies value, while dimensions ('x', 'y', 'z') offer precise measurements. These attributes collectively enable a predictive model for accurate diamond price forecasts, unveiling valuation insights.

# In[3]:


dataframe = pd.read_csv('diamonds.csv')


# # 3.Exploratory Data Analysis :

# Exploratory Data Analysis (EDA) in the "Predicting Diamond Price" project involves comprehensive attribute exploration, starting with statistics for central tendencies and distributions. Visualizations include histograms, box plots, and density plots for numeric attributes, while bar charts depict categorical attribute prevalence. Correlation analysis unveils numeric attribute connections via heatmaps, and scatter plots uncover interactions with the target "price." Grouping, aggregation, and box plots illustrate categorical attribute influences. EDA also covers dimensionality reduction like PCA, outlier handling, missing value checks, and feature transformations. Insights from EDA inform preprocessing, guiding model construction for accurate diamond price prediction.

# In[4]:


dataframe.head()


# In[5]:


dataframe.tail()


# In[6]:


dataframe.shape


# In[9]:


dataframe = dataframe.drop('Unnamed: 0', axis = 1)


# In[10]:


dataframe.head()


# In[11]:


dataframe.shape


# In[12]:


dataframe.info()


# In[61]:


dataframe.isna().sum()


# In[13]:


dataframe.select_dtypes(include = 'object').head()


# In[14]:


print(dataframe['cut'].value_counts(), '\n', dataframe['color'].value_counts(), '\n', dataframe['clarity'].value_counts())


# # 3.1.Data Visualization :

# Data visualization is pivotal in the "Predicting Diamond Price" project's EDA, portraying attributes' distribution through histograms and density plots. Box plots reveal central tendencies and outliers, while bar charts depict categorical attributes like cut, color, and clarity. Correlation heatmaps identify strong relationships, while scatter plots unveil interactions with "price." Box plots by category enable price comparison, 3D scatter plots elucidate dimension relations, and violin plots capture price distributions. Interactive options offer deeper insights. These visualizations guide preprocessing and model construction for precise price predictions.

# # 3.1.1 Single Visualization :

# Single box plot is a powerful visualization for the "Predicting Diamond Price" project's EDA. It showcases how diamond prices differ across categories like "cut," "color," or "clarity." The plot displays median, quartiles, and outliers for each category, providing immediate insights into attribute impact on prices.

# In[34]:


plt.figure(figsize = (10, 5))
sns.histplot(dataframe['price'], bins = 20)


# In[36]:


plt.figure(figsize = (10, 5))
sns.histplot(dataframe['carat'], bins=20)


# In[28]:


plt.figure(figsize=(8, 8))
plt.pie(dataframe['cut'].value_counts(),labels=['Ideal','Premium','Very Good','Good','Fair'],autopct='%1.1f%%')
plt.title('Cut')
plt.show()


# In[23]:


plt.figure(figsize=(5,5))
plt.bar(dataframe['color'].value_counts().index, dataframe['color'].value_counts())
plt.ylabel("Number of Diamonds")
plt.xlabel("Color")
plt.show()


# In[24]:


plt.figure(figsize=(5,5))
plt.bar(dataframe['clarity'].value_counts().index, dataframe['clarity'].value_counts())
plt.title('Clarity')
plt.ylabel("Number of Diamonds")
plt.xlabel("Clarity")
plt.show()


# In[16]:


dataframe["cut"].value_counts().plot.barh().set_title("Class Frequencies of the Cut Variable")


# In[38]:


plt.figure(figsize = (12, 5))
sns.histplot(dataframe['table'], bins=10)
plt.title('Table')
plt.show()


# # 3.1.2 Grouped Visualization :

# Grouped visualizations in the "Predicting Diamond Price" project's EDA unveil attribute interactions within categorical categories like "cut," "color," and "clarity." These visualizations, including box plots, bar charts, scatter plots with hues, and more, offer insights into how attributes influence diamond prices across distinct groups. They provide a nuanced understanding of relationships, aiding data preprocessing and model construction decisions.

# In[40]:


plt.figure(figsize = (12, 5))
sns.barplot(x='cut',
            y='price', 
            data = dataframe)


# In[41]:


plt.figure(figsize = (12, 5))
sns.barplot(x='color',
            y='price',
            data = dataframe)
plt.title('Price vs Color')
plt.show()


# In[26]:


plt.figure(figsize = (12, 5))
sns.barplot(x="cut",
            y="price",
            hue="color",
            data=dataframe)
plt.title("Cut - Price - Color")


# In[43]:


plt.figure(figsize = (12, 5))
sns.barplot(x="cut",
            y="price",
            hue="clarity",
            data = dataframe)
plt.title("Cut - Price - Clarity")


# In[44]:


sns.jointplot(x = "price", 
              y = dataframe["carat"], 
              data = dataframe) 


# # 3.2.Data Preprocessing :

# Data preprocessing is a fundamental step in the "Predicting Diamond Price" project, involving measures like handling missing values, managing outliers, feature engineering, encoding categorical variables, and scaling features. This process enhances dataset quality, ensures compatibility with machine learning algorithms, and forms the basis for accurate predictive modeling.

# In[45]:


dataframe['cut'] = dataframe['cut'].map({'Ideal':5,'Premium':4,'Very Good':3,'Good':2,'Fair':1})

dataframe['color'] = dataframe['color'].map({'D':7,'E':6,'F':5,'G':4,'H':3,'I':2,'J':1})

dataframe['clarity'] = dataframe['clarity'].map({'IF':8,'VVS1':7,'VVS2':6,'VS1':5,'VS2':4,'SI1':3,'SI2':2,'I1':1})


# In[46]:


dataframe.head()


# # 3.2.1 Statical Info :

# Statistical information is pivotal in the "Predicting Diamond Price" project's EDA. Measures like mean, median, and standard deviation offer central tendencies and spread insights. Percentiles, skewness, and kurtosis illuminate data distribution and shape.Statistical insights guide subsequent analysis, visualization, and preprocessing steps.

# In[47]:


dataframe.describe()


# # 3.2.2 Correlation :

# Correlation values highlight how attributes interact, aiding in feature selection, preprocessing, and even new feature creation. Heatmaps visualize correlations, guiding attribute choices and addressing multicollinearity. This analysis informs model development, ensuring accurate prediction of diamond prices.

# In[48]:


corr_dataframe = dataframe.corr()


# In[50]:


corr_dataframe


# In[52]:


plt.figure(figsize = (10, 10))
sns.heatmap(corr_dataframe, 
            annot = True, 
            cmap = 'inferno')


# In[54]:


dataframe.columns


# In[55]:


dataset = dataframe.drop('price', axis = 1)


# In[56]:


dataset.head()


# # 3.2.2.1 Correlation Of Diamond Price with Various Attributes :

# In[57]:


dataset.corrwith(dataframe['price']).plot.bar(
    figsize = (12, 5), 
    title = 'Correlation with Price', 
    cmap = 'ocean'
)


# # 3.2.2.2 Relation Between Price And Caret:

# In[60]:


plt.figure(figsize = (12, 5))
sns.lineplot(x='carat',
             y='price',
             data = dataframe)
plt.title('Carat vs Price')
plt.show()


# In[63]:


columns_to_plot = ['x', 'y', 'z']
titles = ['x vs. carat', 'y vs. carat', 'z vs. carat']

fig, axes = plt.subplots(1, 3, figsize = (15, 5))

for i, column in enumerate(columns_to_plot):
    sns.scatterplot(x = column, y = 'carat', data = dataframe, ax = axes[i])
    axes[i].set_title(titles[i])

plt.tight_layout()
plt.show()


# In[64]:


columns_to_plot = ['x', 'y', 'z']
titles = ['x vs. price', 'y vs. price', 'z vs. price']

fig, axes = plt.subplots(1, 3, figsize = (15, 5))

for i, column in enumerate(columns_to_plot):
    sns.scatterplot(x = column, y='price', data = dataframe, ax = axes[i])
    axes[i].set_title(titles[i])

plt.tight_layout()
plt.show()


# In[65]:


dataframe.head()


# # 3.3.Splitting Dataset :
# 

# Splitting a dataset refers to the process of dividing a given dataset into two or more subsets for training and evaluation purposes. The most common type of split is between the training set and the testing (or validation) set. This division allows us to assess the performance of a machine learning model on unseen data and evaluate its generalization capabilities.
# 
# Train-Test Split: This is the most basic type of split, where the dataset is divided into a training set and a testing set. The training set is used to train the machine learning model, while the testing set is used to evaluate its performance. The split is typically done using a fixed ratio, such as 80% for training and 20% for testing.

# In[66]:


x = dataframe.drop('price', axis = 1)
y = dataframe['price']


# In[67]:


x.shape, y.shape


# In[69]:


from sklearn.model_selection import train_test_split


# In[70]:


x_train, x_test, y_train, y_test = train_test_split(x, 
                                                    y, 
                                                    test_size = 0.2, 
                                                    random_state = 42)


# In[71]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# # 4. Model Selection And Training :

# Model Selection:
# 
# Model selection involves choosing the best algorithm or model architecture for the given problem and dataset. This step requires careful consideration of various factors, such as the nature of the data (e.g., numerical or categorical), the problem type (e.g., regression, classification, clustering), the amount of available data, and the desired model performance. It is essential to select a model that can effectively capture the underlying patterns in the data and make accurate predictions.
# 
# Model Training:
# 
# Once the appropriate model has been selected, the next step is to train it on the dataset. Model training involves adjusting the model's parameters using the training data to make accurate predictions on unseen data. The goal is to minimize the difference between the model's predictions and the actual target values during training.

# In[72]:


from sklearn.tree import DecisionTreeRegressor


# In[73]:


DTR = DecisionTreeRegressor()


# In[74]:


DTR.fit(x_train, y_train)


# In[75]:


DTR.score(x_test, y_test)


# In[76]:


DTR.score(x_train, y_train)


# In[77]:


y_pred_DTR = DTR.predict(x_test)


# In[78]:


y_pred_DTR


# In[93]:


from sklearn.metrics import mean_squared_error,mean_absolute_error


# In[87]:


plt.figure(figsize = (12, 5))
ax = sns.distplot(y_test, hist = False, color = 'r', label = 'Actual Value')
sns.distplot(y_pred_DTR, hist = False, color = 'b',label = 'Fitted Values',ax = ax)
plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price')
plt.ylabel('Proportion of Diamonds')
plt.show()


# In[80]:


from sklearn.ensemble import RandomForestRegressor


# In[81]:


reg = RandomForestRegressor()


# In[82]:


reg.fit(x_train, y_train)


# In[83]:


reg.score(x_test, y_test)


# In[84]:


reg.score(x_train, y_train)


# In[88]:


y_pred_reg = reg.predict(x_test)


# In[89]:


y_pred_reg


# In[91]:


plt.figure(figsize = (12, 5))
ax = sns.distplot(y_test, hist = False, color = 'r',label = 'Actual Value')
sns.distplot(y_pred_reg , hist = False ,color = 'b',label = 'Fitted Values',ax = ax)
plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price')
plt.ylabel('Proportion of Diamonds')
plt.show()


# # Conclusion : 

# Both the models have almost same accuracy. However, the Random Forest Regressor model is slightly better than the Decision Tree Regressor model.

# # Thanks !
