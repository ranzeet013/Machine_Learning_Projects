#!/usr/bin/env python
# coding: utf-8

# # Wine Class Prediction

# The objective of this project is to develop a machine learning model that can accurately predict the class or type of wine based on various input features. The project aims to leverage the power of machine learning algorithms to classify wine samples into different categories based on their characteristics.

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


# In[3]:


import warnings
warnings.filterwarnings('ignore')


# # Importing Dataset

# The project will utilize a dataset containing information about different wine samples. The dataset will include various features such as alcohol content, acidity levels, sugar content, pH levels, and more. Additionally, the dataset will have a target variable that represents the class or type of wine (e.g., red wine, white wine).

# In[4]:


dataframe = pd.read_csv('Wine dataset.csv')


# # Exploratory Data Analysis

# Exploratory Data Analysis is a crucial step in understanding the wine dataset and gaining insights into the relationships between variables. 

# In[5]:


dataframe.head()


# In[6]:


dataframe.tail()


# In[7]:


dataframe.shape


# In[8]:


dataframe.info()


# In[9]:


dataframe['class'].value_counts()


# In[10]:


dataframe['class'].value_counts().plot(kind = 'bar', 
                                       title = 'Different Class of Wine Distribution', 
                                       cmap = 'plasma')


# In[11]:


dataframe.isna().sum().any()


# In[12]:


dataframe.isna().sum()


# # Statical Info 

# In addition to building machine learning models, statistical analysis can provide valuable insights into the wine dataset and help in understanding the relationships between variables. 

# In[13]:


dataframe.describe()


# In[14]:


dataframe.columns


# # Correlation Matrix 

# A correlation matrix is a tabular representation of the correlation coefficients between pairs of variables in a dataset. In the context of the wine class prediction project, constructing a correlation matrix can provide insights into the relationships between different wine features. 

# In[17]:


corr_matrix = dataframe.corr()


# In[18]:


corr_matrix


# In[23]:


plt.figure(figsize = (15, 7))
sns.heatmap(corr_matrix, 
            annot = True, 
            cmap = 'inferno')


# # Correlation of Wine Class with Different Attributes

# In[24]:


dataframe.columns


# In[25]:


dataset = dataframe.drop('class', axis = 1)


# In[26]:


dataset.head()


# In[27]:


dataset.corrwith(dataframe['class']).plot.bar(
    title = 'Correlation With Wine Class', 
    cmap = 'ocean', 
    figsize = (12, 5)
)


# In[28]:


dataframe.head()


# # Splitting Dataset

# In the wine class prediction project, it is important to split the dataset into separate subsets for training and testing the machine learning models. This allows us to assess the model's performance on unseen data and evaluate its ability to generalize to new wine samples.

# Train-Test Split: This is the most basic type of split, where the dataset is divided into a training set and a testing set. The training set is used to train the machine learning model, while the testing set is used to evaluate its performance. The split is typically done using a fixed ratio, such as 80% for training and 20% for testing.

# In[29]:


x = dataframe.drop('class', axis = 1)
y = dataframe['class']


# In[30]:


x.shape, y.shape


# In[31]:


from sklearn.model_selection import train_test_split


# In[32]:


x_train, x_test, y_train, y_test = train_test_split(x, 
                                                    y, 
                                                    test_size = 0.2, 
                                                    random_state = 42)


# In[33]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# # Scaling

# Scaling is an essential preprocessing step in the wine class prediction project, as it helps ensure that the numerical features are on a similar scale. Scaling is particularly important when using machine learning algorithms that are sensitive to the magnitude of input features.

# In[34]:


from sklearn.preprocessing import StandardScaler


# In[35]:


scaler = StandardScaler()


# In[36]:


x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[37]:


x_train


# In[38]:


x_test


# In[39]:


x_train.shape, x_test.shape


# # Model Training

# Model training is a crucial step in the wine class prediction project, where machine learning algorithms learn from the labeled data to make accurate predictions on unseen wine samples.

# In[41]:


from sklearn.ensemble import RandomForestClassifier


# In[42]:


model = RandomForestClassifier()


# In[43]:


model.fit(x_train, y_train)


# In[44]:


y_pred = model.predict(x_test)


# In[45]:


y_pred


# In[46]:


model.score(x_train, y_train)


# In[47]:


model.score(x_test, y_test)


# In[48]:


print(y_test[12]), print(y_pred[12])


# # Error Analysis

# Error analysis is a crucial step in understanding the performance of machine learning models and identifying areas for improvement in the wine class prediction project. It involves analyzing and interpreting the errors made by the models during prediction.

# In[54]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[55]:


accuracy_score = accuracy_score(y_test, y_pred)


# In[56]:


accuracy_score


# In[57]:


confusion_matrix = confusion_matrix(y_test, y_pred)


# In[58]:


confusion_matrix


# In[59]:


plt.figure(figsize = (6, 4))
sns.heatmap(confusion_matrix, 
            annot = True, 
            cmap = 'RdPu')


# In[61]:


print(classification_report(y_test, y_pred))


# By conducting error analysis, you can gain a deeper understanding of the models' performance, identify error patterns, and develop strategies to improve the wine class prediction. Error analysis helps refine the models, enhance their accuracy, and make more informed decisions in the wine industry based on the models' predictions.

# # Thanks !
