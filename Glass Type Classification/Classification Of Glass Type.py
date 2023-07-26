#!/usr/bin/env python
# coding: utf-8

# # Classification Of Glass Type

# The Glass Type Classification project is a machine learning task aimed at predicting the type or category of glass samples based on their chemical composition. The dataset used for this project contains various attributes, such as Refractive Index (RI), Sodium (Na), Magnesium (Mg), Aluminum (Al), Silicon (Si), Potassium (K), Calcium (Ca), Barium (Ba), and Iron (Fe), which represent the percentage of different oxides in each glass sample.

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

# The glass type dataset contains information about different glass samples, such as Refractive Index (RI), and the percentages of various chemical components including Sodium (Na), Magnesium (Mg), Aluminum (Al), Silicon (Si), Potassium (K), Calcium (Ca), Barium (Ba), and Iron (Fe). The dataset also includes a "Type" column representing the class or type of each glass sample. This dataset is commonly used for machine learning tasks to predict the glass type based on its chemical composition, which can have practical applications in various industries.

# In[3]:


dataframe = pd.read_csv('glass.csv')


# # Exploratory Data Analysis

#  Exploratory Data Analysis is the process of examining and visualizing data to gain insights, understand patterns, detect anomalies, and formulate hypotheses. EDA involves using various statistical and graphical techniques to summarize and explore the main characteristics of a dataset. It helps in understanding the data's distribution, relationships between variables, and potential trends or outliers. EDA plays a crucial role in data preprocessing and preparation before applying machine learning algorithms or making data-driven decisions.

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


dataframe.duplicated().sum()


# In[11]:


dataframe.drop_duplicates(inplace=True)


# In[12]:


dataframe.columns


# In[13]:


dataframe['Type'].value_counts()


# # Distribution Of Different Types Of Glasses

# In[14]:


type_counts = dataframe['Type'].value_counts()
plt.figure(figsize=(12, 5))
plt.bar(type_counts.index, type_counts.values, width=0.5)
plt.title('Distribution of Glass Type')
plt.xlabel('Glass Type')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()


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


# In[19]:


dataframe.columns


# In[20]:


dataset = dataframe.drop('Type', axis = 1)


# In[21]:


dataset.head()


# In[22]:


dataset.corrwith(dataframe['Type']).plot.bar(
    title = 'Correlation With Glass Type', 
    figsize = (12, 5), 
    rot = 90, 
    cmap = 'plasma'
)


# In[23]:


dataframe.columns


# # Splitting Dataset

# Splitting a dataset refers to the process of dividing a given dataset into two or more subsets for training and evaluation purposes. The most common type of split is between the training set and the testing (or validation) set. This division allows us to assess the performance of a machine learning model on unseen data and evaluate its generalization capabilities.
# 
# Train-Test Split: This is the most basic type of split, where the dataset is divided into a training set and a testing set. The training set is used to train the machine learning model, while the testing set is used to evaluate its performance. The split is typically done using a fixed ratio, such as 80% for training and 20% for testing

# In[24]:


x = dataframe.drop('Type', axis = 1)
y = dataframe['Type']


# In[25]:


x.shape, y.shape


# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


x_train, x_test, y_train, y_test = train_test_split(x, 
                                                    y, 
                                                    test_size = 0.2, 
                                                    random_state = 42)


# In[28]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# # Model Selection And Training 

# Model selection is the process of choosing the most suitable machine learning algorithm for a specific task by comparing and evaluating different models' performance. Model training involves feeding the selected model with training data, allowing it to learn patterns and relationships within the data. During training, the model's internal parameters are adjusted iteratively to minimize the difference between its predictions and the actual target values in the training data. The trained model can then be used to make predictions on new, unseen data.

# In[30]:


from sklearn.ensemble import RandomForestClassifier


# In[31]:


clf = RandomForestClassifier(n_estimators= 500, 
                            criterion= 'entropy', 
                            max_depth= None, 
                            min_samples_split = 2, 
                            min_samples_leaf = 1, 
                            max_features = 'sqrt' )


# In[32]:


clf.fit(x_train, y_train)


# In[33]:


y_pred = clf.predict(x_test)


# In[34]:


y_pred


# # Error Analysis

# 
# Error analysis is a crucial step in the evaluation and improvement of machine learning models. It involves the systematic examination and understanding of the errors made by the model during prediction. The primary goal of error analysis is to identify patterns and sources of mistakes made by the model, which can provide valuable insights into its performance and guide improvements.

# Recall score, precision score, accuracy score, classification score, and confusion matrix are vital evaluation metrics in classification models. Recall measures the ability to correctly detect positive instances, while precision measures the accuracy of positive predictions. Accuracy gauges overall correctness, but it may be misleading in imbalanced datasets. The classification score is a comprehensive view of multiple metrics. The confusion matrix visually presents model predictions against actual labels, helping calculate TP, TN, FP, and FN. These metrics and the confusion matrix offer valuable insights into model performance, guiding decisions for improvement and optimization.

# In[39]:


from sklearn.metrics import recall_score, precision_score, accuracy_score, classification_report, confusion_matrix


# In[40]:


accuracy_score = accuracy_score(y_pred, y_test)
print('Accuracy Score:', accuracy_score)


# In[42]:


precision = precision_score(y_test, y_pred, average='weighted')
print("Precision Score:", precision)


# In[43]:


recall_score = recall_score(y_pred, y_test, average = 'weighted')
print('Recall Score :', recall_score)


# In[46]:


print(classification_report(y_pred, y_test))


# In[47]:


confusion_matrix = confusion_matrix(y_pred, y_test)


# In[48]:


confusion_matrix


# In[49]:


plt.figure(figsize = (6, 4))
sns.heatmap(confusion_matrix, 
            annot = True, 
            cmap = 'inferno')


# # Thanks !
