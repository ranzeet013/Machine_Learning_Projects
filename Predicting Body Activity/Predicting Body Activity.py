#!/usr/bin/env python
# coding: utf-8

# # Predicting Body Activity

# The objective of this project is to develop machine learning models that can predict and analyze body activity based on various input data. By leveraging machine learning techniques, we aim to create models that can accurately predict and classify different body activities, providing valuable insights and potential applications in areas such as healthcare, fitness tracking, and sports performance analysis.

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

# In[4]:


train = pd.read_csv("train-1.csv")
test = pd.read_csv("test.csv")


# # Exploratory Data Analysis

# Before diving into building machine learning models, it is essential to perform exploratory data analysis (EDA) to gain a better understanding of the dataset and its characteristics. EDA helps in identifying patterns, trends, anomalies, and relationships within the data, which can guide the preprocessing and modeling stages of the project. 

# In[5]:


train.head()


# In[6]:


test.head()


# In[7]:


train.tail()


# In[8]:


test.tail()


# In[9]:


train.shape


# In[10]:


test.shape


# It is important to note that joining datasets should be done cautiously, considering the compatibility and requirements of the analysis or task at hand. It is generally recommended to maintain the original train and test dataset separation for proper model development and evaluation.

# In[12]:


train['Data'] = 'Train'
test['Data'] = 'Test'
dataframe = pd.concat([train, test], axis=0).reset_index(drop=True)
dataframe['subject'] = '#' + both['subject'].astype(str)


# In[13]:


dataframe.head()


# In[14]:


dataframe.shape


# In[15]:


dataframe.columns


# In[17]:


def basic_details(df):
    b = pd.DataFrame()
    b['Missing value'] = df.isnull().sum()
    b['N unique value'] = df.nunique()
    b['dtype'] = df.dtypes
    return b
basic_details(dataframe)


# Data visualization is a crucial step in exploratory data analysis (EDA) as it allows you to gain insights, identify patterns, and communicate findings effectively.

# In[18]:


activity = both['Activity']
label_counts = activity.value_counts()

plt.figure(figsize= (12, 8))
plt.bar(label_counts.index, label_counts)


# In[19]:


Data = both['Data']
Subject = both['subject']
train = both.copy()
train = train.drop(['Data','subject','Activity'], axis =1)


# # Scaling 

# Scaling is a preprocessing technique used in machine learning to transform numeric features to a common scale. It ensures that all features contribute equally to the analysis and modeling process, preventing features with larger magnitudes from dominating the results. 

# In[20]:


from sklearn.preprocessing import StandardScaler
slc = StandardScaler()
train = slc.fit_transform(train)


# In[21]:


from sklearn.decomposition import PCA
pca = PCA(n_components=0.9, random_state=0)
train = pca.fit_transform(train)


# # Splitting Dataset

# Splitting a dataset refers to dividing it into separate subsets for training, validation, and testing purposes. This division allows us to assess the performance of machine learning models on unseen data and avoid overfitting. Here are the commonly used dataset splitting techniques:

# Train-Test Split:
# 
# The train-test split involves dividing the dataset into two subsets: a training set and a testing set.
# 
# The training set is used to train the machine learning model, while the testing set is used to evaluate its performance on unseen data.
# 
# The typical split ratio is around 70-80% for training and 20-30% for testing, depending on the size of the dataset and the available data.
# 
# The train-test split is suitable when there is sufficient data available, and the focus is on model evaluation.

# In[22]:


from sklearn.model_selection import train_test_split


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(train, activity, test_size = 0.2, random_state = 0)


# In[24]:


num_folds = 10
seed = 0
scoring = 'accuracy'
results = {}
accuracy = {}


# # Model Selection And Training

# Model selection and training are crucial steps in machine learning projects. The goal is to choose an appropriate model architecture or algorithm and train it on the available dataset. 

# In[27]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import KFold, cross_val_score
model = KNeighborsClassifier(algorithm= 'auto', n_neighbors= 8, p= 1, weights= 'distance')

_ = cross_val_score(model, X_train, y_train, cv=10, scoring=scoring)
results["GScv"] = (_.mean(), _.std())

model.fit(X_train, y_train) 
y_predict = model.predict(X_test)

accuracy["GScv"] = accuracy_score(y_test, y_predict)

print(classification_report(y_test, y_predict))

cm= confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot=True)


# # Thanks !
