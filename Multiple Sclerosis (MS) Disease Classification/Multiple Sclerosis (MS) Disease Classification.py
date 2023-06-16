#!/usr/bin/env python
# coding: utf-8

# #  Multiple Sclerosis (MS) Disease Classification 

# The goal of this project is to develop a machine learning model for the classification of Multiple Sclerosis (MS) disease. MS is a chronic autoimmune disease affecting the central nervous system, causing various neurological symptoms. Early detection and accurate classification of MS can assist in timely intervention and treatment planning.

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


# # Reading Dataset

# In[3]:


dataframe = pd.read_csv('conversion_predictors_of_clinically_isolated_syndrome_to_multiple_sclerosis.csv')


# # Exploratory Data Analysis

# The process of analyzing and understanding a dataset to gain insights and identify patterns or trends. The goal of exploring the data is to become familiar with its structure, distribution, and quality, as well as to identify potential issues or anomalies that may need to be addressed before further analysis.

# In[4]:


dataframe.head()


# In[5]:


dataframe.tail()


# In[6]:


dataframe = dataframe.drop('Unnamed: 0', axis = 1)


# In[7]:


dataframe.head()


# In[8]:


dataframe.shape


# In[9]:


dataframe.info()


# # Data Cleaning

# Data cleaning is an essential step in the data preprocessing phase of a machine learning project. It involves identifying and handling missing values, dealing with outliers, removing duplicates, and addressing other data quality issues

# Remove rows or columns with missing values if they are few or have little impact on the overall dataset.
# 
# Impute missing values using techniques like mean, median

# In[10]:


dataframe.isna().sum().any()


# In[11]:


dataframe.isna().sum()


# In[12]:


dataframe.columns


# In[13]:


dataframe = dataframe.drop(['Initial_EDSS', 'Final_EDSS'], axis = 1)


# In[14]:


dataframe.head()


# In[15]:


dataframe['Schooling'] = dataframe['Schooling'].fillna(dataframe['Schooling'].mean())


# In[16]:


dataframe['Initial_Symptom'] = dataframe['Initial_Symptom'].fillna(dataframe['Initial_Symptom'].mean())


# In[17]:


dataframe.head()


# In[18]:


dataframe.isna().sum().any()


# # Data Visualization

# Data visualization is a powerful technique for gaining insights, identifying patterns, and communicating findings from data. It involves creating visual representations of data using charts, graphs, plots, and other graphical elements.

# In[19]:


dataframe['group'].value_counts()


# In[20]:


dataframe['group'].value_counts().plot(kind = 'bar', 
                                       cmap = 'plasma', 
                                       figsize = (10, 4), 
                                       title = 'Group with sclerosis', 
                                       rot = 90)


# In[21]:


dataframe['Gender'].value_counts()


# In[22]:


pd.crosstab(dataframe.Gender, dataframe.group)


# In[23]:


pd.crosstab(dataframe.Gender, dataframe.group).plot(kind = 'bar',
                                                    cmap = 'plasma',
                                                    figsize = (12, 5), 
                                                    title = 'Gender with Sclerosis syndrom', 
                                                    rot = 90)


# # Statical Info

# Statistical information refers to numerical data or metrics that describe various aspects of a dataset or population. These statistics provide quantitative measures of central tendency, dispersion, relationships, and other properties of the data.

# In[24]:


dataframe.describe()


# # Correlation Matrix

# A correlation matrix is a table that shows the pairwise correlations between variables in a dataset. Each cell in the table represents the correlation between two variables, and the strength and direction of the correlation is indicated by the color and magnitude of the cell.
# 
# Correlation matrices are commonly used in data analysis to identify relationships between variables and to help understand the structure of the data. The values in the correlation matrix range from -1 to 1, with -1 indicating a perfect negative correlation, 1 indicating a perfect positive correlation, and 0 indicating no correlation.

# In[25]:


corr_matrix = dataframe.corr()


# In[26]:


corr_matrix


# In[27]:


plt.figure(figsize = (18, 10))
sns.heatmap(corr_matrix, 
            annot = True, 
            cmap = 'inferno')


# In[28]:


dataset = dataframe.drop('group', axis = 1)


# In[29]:


dataset.head()


# In[30]:


dataset.corrwith(dataframe['group']).plot.bar(
    figsize = (12, 5), 
    title = 'Correlation With Sclerosis Syndrom Group', 
    rot = 90, 
    cmap = 'seismic'
)


# In[31]:


dataframe.columns


# # Splitting Dataset

# Splitting a dataset refers to the process of dividing a given dataset into two or more subsets for training and evaluation purposes. The most common type of split is between the training set and the testing (or validation) set. This division allows us to assess the performance of a machine learning model on unseen data and evaluate its generalization capabilities.
# 
# Train-Test Split: This is the most basic type of split, where the dataset is divided into a training set and a testing set. The training set is used to train the machine learning model, while the testing set is used to evaluate its performance. The split is typically done using a fixed ratio, such as 80% for training and 20% for testing.

# In[32]:


x = dataframe.drop('group', axis = 1)
y = dataframe['group']


# In[33]:


x.shape, y.shape


# In[34]:


from sklearn.model_selection import train_test_split


# In[35]:


x_train, x_test, y_train, y_test = train_test_split(x, 
                                                    y, 
                                                    test_size = 0.2, 
                                                    random_state = 101)


# In[36]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# # Scaling

# Scaling is a preprocessing technique used in machine learning to transform the input features to a similar scale. It is often necessary because features can have different units, ranges, or magnitudes, which can affect the performance of certain algorithms. Scaling ensures that all features contribute equally to the learning process and prevents features with larger values from dominating those with smaller values.
# 
# StandardScaler is a commonly used method for scaling numerical features in machine learning. It is part of the preprocessing module in scikit-learn, a popular machine learning library in Python.

# In[37]:


from sklearn.preprocessing import StandardScaler


# In[38]:


scaler = StandardScaler()


# In[39]:


x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[40]:


x_train


# In[41]:


x_test


# In[42]:


x_train.shape, x_test.shape


# # Machine Learning Models

# There are numerous machine learning models available for various tasks, including classification, regression, clustering, and more. Here are some commonly used machine learning models:

# Random Forest: An ensemble model that combines multiple decision trees to improve accuracy and reduce overfitting.

# In[46]:


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# In[47]:


random_forest = RandomForestClassifier()


# In[48]:


random_forest.fit(x_train, y_train)


# In[49]:


y_pred_random = random_forest.predict(x_test)


# In[50]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, r2_score


# # Error Analysis

# Error analysis is an important step in evaluating and improving the performance of a machine learning model. It involves analyzing the errors made by the model during prediction or classification tasks and gaining insights into the types of mistakes it is making. Error analysis can provide valuable information for model refinement and identifying areas for improvement

# # Classification Report

# The classification report is a performance evaluation metric for classification models that provides various metrics such as precision, recall, F1-score, and support for each class. These metrics help assess the model's performance in terms of class-wise classification accuracy.

# # Accuracy Score

# Accuracy is a commonly used metric for evaluating classification models. It measures the proportion of correctly classified instances out of the total instances. The formula for calculating accuracy is as follows:
# 
# Accuracy = (Number of Correct Predictions) / (Total Number of Predictions)

# # Confusion Matrix

# A confusion matrix is a table that summarizes the performance of a classification model by showing the counts of true positives, true negatives, false positives, and false negatives. The confusion matrix provides valuable insights into the model's performance for each class and can be used to calculate various evaluation metrics. 

#                  Predicted
#                |  Class 0 | Class 1 |
# ------------------------------------
# 
# Actual Class 0 |    TN    |   FP    |
# 
# Actual Class 1 |    FN    |   TP    |
# 

# # r2 Score

# The R2 score (also known as the coefficient of determination) is a commonly used metric for evaluating regression models. It measures the proportion of the variance in the dependent variable that is predictable from the independent variables. The formula for calculating the R2 score is as follows:
# 
# R2 Score = 1 - (Sum of Squared Residuals / Total Sum of Squares)

# In[51]:


acc_random = accuracy_score(y_test, y_pred_random)


# In[53]:


acc_random


# In[54]:


print(classification_report(y_test, y_pred_random))


# In[56]:


r2_random = r2_score(y_test, y_pred_random)


# In[57]:


r2_random


# In[55]:


confusion_matrix_random = confusion_matrix(y_test, y_pred_random)


# In[58]:


plt.figure(figsize = (6, 4))
sns.heatmap(confusion_matrix_random, 
            annot = True, 
            cmap = 'inferno')


# Support Vector Machines (SVM): A powerful classification model that separates data using hyperplanes to maximize the margin between different classes.

# In[61]:


svc = SVC()


# In[62]:


svc.fit(x_train, y_train)


# In[63]:


y_pred_svc = svc.predict(x_test)


# In[64]:


accuracy_score_svc = accuracy_score(y_test, y_pred_svc)


# In[65]:


accuracy_score_svc


# In[66]:


print(classification_report(y_test, y_pred_svc))


# In[67]:


r2_score_svc = r2_score(y_test, y_pred_svc)


# In[68]:


r2_score_svc


# In[69]:


confusion_matrix_svc = confusion_matrix(y_test, y_pred_svc)


# In[70]:


plt.figure(figsize = (6, 4))
sns.heatmap(confusion_matrix_svc, 
            annot = True, 
            cmap = 'inferno')

