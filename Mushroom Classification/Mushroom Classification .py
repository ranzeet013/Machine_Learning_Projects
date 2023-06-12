#!/usr/bin/env python
# coding: utf-8

# # Mushroom Classification with ML Algorithms

# The Mushroom Classification with ML Algorithm project aims to develop a system that can accurately classify different types of mushrooms based on their characteristics. The project utilizes machine learning algorithms to train a model on a dataset of mushroom samples and their corresponding labels.

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


# # Importing Dataset

# In[2]:


dataframe = pd.read_csv('mushrooms.csv')


# # Exploratory Data Analysis

# The process of analyzing and understanding a dataset to gain insights and identify patterns or trends. The goal of exploring the data is to become familiar with its structure, distribution, and quality, as well as to identify potential issues or anomalies that may need to be addressed before further analysis.

# In[3]:


dataframe.head()


# In[4]:


dataframe.tail()


# # Encoding 

# Encoding refers to the process of transforming categorical variables into numerical representations that can be easily understood by machine learning algorithms. Categorical variables are variables that take on a limited set of distinct values, such as colors (red, blue, green), sizes (small, medium, large), or categories (A, B, C).

# Label Encoding: This technique assigns a unique numerical value to each category in a categorical variable. For example, if we have a variable "color" with categories "red," "blue," and "green," label encoding would assign them numeric values like 0, 1, and 2. However, label encoding may introduce an implicit ordinal relationship between the categories, which may not be desired in some cases

# In[5]:


from sklearn.preprocessing import LabelEncoder


# In[6]:


encoder = LabelEncoder()


# In[7]:


for column in dataframe.columns:
    dataframe[column] = encoder.fit_transform(dataframe[column])


# In[8]:


dataframe.head()


# In[9]:


dataframe.tail()


# In[10]:


dataframe.info()


# In[11]:


dataframe.isna().sum().any()


# In[12]:


dataframe.isna().sum()


# In[13]:


dataframe.shape


# In[14]:


dataframe.columns


# In[15]:


dataframe['class'].describe()


# In[16]:


dataframe['class'].value_counts()


# In[17]:


dataframe['class'].value_counts().plot(kind = 'bar', 
                                       figsize = (10, 4), 
                                       title = 'Mushroom Class', 
                                       cmap = 'magma')


# # Dataset Profile Report

# A data profile report provides a comprehensive summary and analysis of a dataset, offering insights into its structure, characteristics, and statistical properties. It helps in understanding the data, identifying potential issues, and making informed decisions during the data preprocessing and analysis phases.

# In[18]:


from pandas_profiling import ProfileReport


# In[19]:


profile_report = ProfileReport(dataframe, minimal = True)
profile_report.to_file('Mushroom_Profile_Report.html')
profile_report


# # Statical Info

# Statistical information refers to numerical data or metrics that describe various aspects of a dataset or population. These statistics provide quantitative measures of central tendency, dispersion, relationships, and other properties of the data.

# In[20]:


dataframe.describe()


# # Correlation Matrix

# A correlation matrix is a table that shows the pairwise correlations between variables in a dataset. Each cell in the table represents the correlation between two variables, and the strength and direction of the correlation is indicated by the color and magnitude of the cell.
# 
# Correlation matrices are commonly used in data analysis to identify relationships between variables and to help understand the structure of the data. The values in the correlation matrix range from -1 to 1, with -1 indicating a perfect negative correlation, 1 indicating a perfect positive correlation, and 0 indicating no correlation.

# In[21]:


corr_matrix = dataframe.corr()


# In[22]:


corr_matrix


# In[23]:


plt.figure(figsize = (24, 14))
sns.heatmap(corr_matrix, 
            annot = True, 
            cmap = 'coolwarm')


# In[24]:


dataframe.columns


# In[25]:


dataset = dataframe.drop('class', axis = 1)


# In[26]:


dataset.head()


# In[27]:


dataset.corrwith(dataframe['class']).plot.bar(
    figsize = (12, 6), 
    title = 'Correlation with Mushroom Class', 
    rot = 90
)


# # Splitting Dataset

# Splitting a dataset refers to the process of dividing a given dataset into two or more subsets for training and evaluation purposes. The most common type of split is between the training set and the testing (or validation) set. This division allows us to assess the performance of a machine learning model on unseen data and evaluate its generalization capabilities.
# 
# Train-Test Split: This is the most basic type of split, where the dataset is divided into a training set and a testing set. The training set is used to train the machine learning model, while the testing set is used to evaluate its performance. The split is typically done using a fixed ratio, such as 80% for training and 20% for testing.

# In[28]:


dataframe.head()


# In[29]:


x = dataframe.drop('class', axis = 1)
y = dataframe['class']


# In[30]:


from sklearn.model_selection import train_test_split


# In[31]:


x_train, x_test, y_train, y_test = train_test_split(x, 
                                                    y, 
                                                    test_size = 0.2, 
                                                    random_state = 42)


# In[32]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# # Scaling

# Scaling is a preprocessing technique used in machine learning to transform the input features to a similar scale. It is often necessary because features can have different units, ranges, or magnitudes, which can affect the performance of certain algorithms. Scaling ensures that all features contribute equally to the learning process and prevents features with larger values from dominating those with smaller values.
# 
# StandardScaler is a commonly used method for scaling numerical features in machine learning. It is part of the preprocessing module in scikit-learn, a popular machine learning library in Python.

# In[33]:


from sklearn.preprocessing import StandardScaler


# In[34]:


scaler = StandardScaler()


# In[35]:


x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[36]:


x_train


# In[37]:


x_test


# In[38]:


x_train.shape, x_test.shape


# # Random Forest Classifier

# A Random Forest Classifier is a popular machine learning algorithm that combines the principles of ensemble learning and decision trees. It is used for classification tasks, where the goal is to predict the class or category of a given input based on its features.

# # Error Analysis

# Error analysis is an important step in evaluating and improving the performance of a machine learning model. It involves analyzing the errors made by the model during prediction or classification tasks and gaining insights into the types of mistakes it is making. Error analysis can provide valuable information for model refinement and identifying areas for improvement

# A confusion matrix is a table that summarizes the performance of a classification model by showing the counts of true positive (TP), true negative (TN), false positive (FP), and false negative (FN) predictions. It is a useful tool for evaluating the accuracy and effectiveness of a classification model.

# A classification report is a summary of various evaluation metrics for a classification model. It provides a comprehensive overview of the model's performance, including metrics such as precision, recall, F1 score, and support.

# Accuracy score is a commonly used metric to evaluate the performance of a classification model. It measures the proportion of correct predictions made by the model out of the total number of predictions.
# 
# The accuracy score is calculated using the following formula:
# 
# Accuracy = (Number of Correct Predictions) / (Total Number of Predictions)

# In[39]:


from sklearn.ensemble import RandomForestClassifier
random_classifier = RandomForestClassifier()
random_classifier.fit(x_train, y_train)


# In[40]:


y_pred_random = random_classifier.predict(x_test)


# In[42]:


from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# In[43]:


accuracy_score_random = accuracy_score(y_test, y_pred_random)


# In[45]:


accuracy_score_random


# In[46]:


print(classification_report(y_test, y_pred_random))


# In[48]:


confusion_matrix_random = confusion_matrix(y_test, y_pred_random)


# In[51]:


plt.figure(figsize = (6, 4))
sns.heatmap(confusion_matrix_random, 
            annot = True, 
            cmap = 'RdPu')


# # Logistic Regression

# Logistic regression is a statistical modeling technique used for binary classification problems, where the goal is to predict a binary outcome variable based on a set of input features. It is a type of regression analysis that models the relationship between the predictors (input features) and the probability of the binary outcome.

# In[52]:


from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)


# In[53]:


y_pred_log_reg = log_reg.predict(x_test)


# In[54]:


accuracy_score_log_reg = accuracy_score(y_test, y_pred_log_reg)


# In[55]:


accuracy_score_log_reg


# In[56]:


print(classification_report(y_test, y_pred_log_reg))


# In[57]:


confusion_matrix_log_reg = confusion_matrix(y_test, y_pred_log_reg)


# In[58]:


plt.figure(figsize = (6, 4))
sns.heatmap(confusion_matrix_log_reg, 
            annot = True, 
            cmap = 'RdPu')


# # Decision Tree Classifier

# A Decision Tree Classifier is a machine learning algorithm that uses a tree-like model to make predictions based on a set of input features. It is a popular algorithm for both classification and regression tasks, but here we will focus on its application as a classifier.

# In[60]:


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)


# In[61]:


y_pred_tree = tree.predict(x_test)


# In[62]:


accuracy_score_tree = accuracy_score(y_test, y_pred_tree)


# In[63]:


accuracy_score_tree


# In[65]:


print(classification_report(y_test, y_pred_tree))


# In[66]:


confusion_matrix_tree = confusion_matrix(y_test, y_pred_tree)


# In[67]:


plt.figure(figsize = (6, 4))
sns.heatmap(confusion_matrix_tree, 
            annot = True, 
            cmap = 'RdPu')


# # KNeighbors Classifier

# The K-Nearest Neighbors (KNN) Classifier is a machine learning algorithm used for both classification and regression tasks. It is a non-parametric algorithm that makes predictions based on the similarity of a new data point to its neighboring data points in the training dataset.

# In[69]:


from sklearn.neighbors import KNeighborsClassifier
knc = KNeighborsClassifier()
knc.fit(x_train, y_train)


# In[70]:


y_pred_knc = knc.predict(x_test)


# In[71]:


accuracy_score_knc = accuracy_score(y_test, y_pred_knc)


# In[72]:


accuracy_score_knc


# In[73]:


print(classification_report(y_test, y_pred_knc))


# In[74]:


confusion_matrix_knc = confusion_matrix(y_test, y_pred_knc)


# In[75]:


plt.figure(figsize = (6, 4))
sns.heatmap(confusion_matrix_knc, 
            annot = True, 
            cmap = 'RdPu')


# # SVC

# SVC stands for Support Vector Classifier, which is a machine learning algorithm used for binary classification tasks. It is based on the principles of Support Vector Machines (SVM) and is an extension of SVM for classification purposes.

# In[77]:


from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)


# In[78]:


y_pred_svc = svc.predict(x_test)


# In[79]:


accuracy_score_svc = accuracy_score(y_test, y_pred_svc)


# In[80]:


accuracy_score_svc


# In[81]:


print(classification_report(y_test, y_pred_svc))


# In[83]:


confusion_matrix_svc = confusion_matrix(y_test, y_pred_svc)


# In[84]:


plt.figure(figsize = (6, 4))
sns.heatmap(confusion_matrix_svc, 
            annot = True, 
            cmap = 'RdPu')


# # Accuracy Table

# In[87]:


index = ["Random Forest Classifier", "Logistic Regression", "Decision Tree Classifier", "KNN", "SVC"]
data = {"Accuracy Score": [1.00, 0.95, 1.00, 1, 1]}

prediction_result = pd.DataFrame(data, index=index)
prediction_result.index.name = "Algorithms Used"


# In[88]:


print(prediction_result)

