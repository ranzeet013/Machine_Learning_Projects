#!/usr/bin/env python
# coding: utf-8

# # Predicting Cervical Cancer :

# Cervical cancer is one of the leading causes of cancer-related deaths among women worldwide. Early detection and accurate prediction of cervical cancer can significantly improve the chances of successful treatment and save lives. This project aims to develop a predictive model using machine learning techniques to identify individuals at high risk of cervical cancer, allowing for timely intervention and medical care. 

# The main objective of this project is to build a robust machine learning model that can predict the likelihood of cervical cancer based on relevant features and patient data. By analyzing a dataset of cervical cancer cases, the model will learn patterns and correlations to make accurate predictions, enabling healthcare professionals to identify high-risk patients.

# # Import Libraries :

# These are just a few examples of popular Python libraries. You can import any other library using the same import statement followed by the library name or alias:
# 
# NumPy: for numerical operations and array manipulation
# 
# Pandas: for data manipulation and analysis
# 
# Matplotlib: for creating visualizations
# 
# Scikit-learn: for machine learning algorithms.

# In[25]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# # Importing Dataset :

# The dataset contains various patient attributes, including demographic data, sexual behavior, smoking habits, contraceptive usage, history of sexually transmitted diseases (STDs), and diagnostic outcomes related to cervical cancer. These attributes will be used to develop a machine learning model for predicting the risk of cervical cancer, facilitating early detection and timely intervention.

# In[2]:


cancer_dataframe = pd.read_csv('cervical-cancer_csv.csv')


# # Exploratory Data Analysis :

# Exploratory Data Analysis (EDA) for the cervical cancer dataset involves summarizing key statistics, visualizing data distributions, and identifying relationships between variables. This includes handling missing data, analyzing target variable distribution, checking for class imbalances, and detecting outliers. Correlation analysis helps understand feature relationships, and feature importance techniques can identify influential predictors. EDA insights guide data preprocessing steps like feature scaling and encoding, preparing the dataset for building a predictive model to identify cervical cancer risk accurately.

# In[3]:


cancer_dataframe.head()


# In[4]:


cancer_dataframe.tail()


# In[5]:


cancer_dataframe.shape


# In[6]:


cancer_dataframe.info()


# In[7]:


cancer_dataframe.isna().any()


# In[14]:


plt.figure(figsize = (12, 6))
sns.heatmap(cancer_dataframe.isna(), yticklabels = False)
plt.show()


# In[8]:


cancer_dataframe = cancer_dataframe.replace('?', np.nan)


# In[9]:


cancer_dataframe


# In[11]:


cancer_dataframe.describe()


# In[15]:


cancer_dataframe.mean()


# In[16]:


cancer_dataframe  = cancer_dataframe.fillna(cancer_dataframe.mean())


# In[17]:


cancer_dataframe.isna().any()


# # Statical Info:

# Statistical information in the Exploratory Data Analysis (EDA) of the cervical cancer dataset includes calculating summary statistics like mean, median, and standard deviation for numerical attributes. Visualization of data distributions through histograms and kernel density plots provides insights into data spread. Frequency distribution of categorical variables reveals the distribution of different categories.

# In[18]:


cancer_dataframe.describe()


# # Correlation Matrix :

# A correlation matrix is a square matrix showing the correlation coefficients between pairs of numerical variables in the cervical cancer dataset. It helps reveal the strength and direction of relationships between attributes, ranging from -1 to 1. A value of 1 indicates a perfect positive correlation, -1 represents a perfect negative correlation, and 0 implies no correlation. By analyzing the correlation matrix, we can identify potential dependencies and interactions among variables, assisting in feature selection and multicollinearity detection for building an accurate predictive model.

# In[19]:


corr_matrix = cancer_dataframe.corr()


# In[21]:


corr_matrix


# In[23]:


plt.figure(figsize = (30, 30))
sns.heatmap(corr_matrix, 
            annot = True, 
            cmap = 'inferno')


# In[24]:


cancer_dataframe.hist(bins = 10, figsize = (30,30), color='blue')
plt.show()


# In[26]:


cancer_dataframe.columns


# # Correlation Dataframe :

# In[27]:


dataframe  = cancer_dataframe.drop('Biopsy', axis = 1)


# In[28]:


dataframe.head()


# In[30]:


dataframe.corrwith(cancer_dataframe['Biopsy']).plot.bar(
    title = 'Correlation With Biopsy', 
    figsize = (12, 5), 
    cmap = 'plasma'
)


# In[31]:


cancer_dataframe.head()


# In[32]:


input_df = cancer_dataframe.drop('Biopsy', axis = 1)
target_df = cancer_dataframe['Biopsy']


# In[35]:


x = np.array(input_df).astype('float32')
y = np.array(target_df).astype('float32')


# In[36]:


x.shape, y.shape


# In[37]:


y.reshape(-1, 1)


# # Scaling :

# Scaling is a preprocessing step in machine learning that involves transforming the features or variables of your dataset to a consistent scale. It is important because many machine learning algorithms are sensitive to the scale of the input features. Scaling helps ensure that all features have a similar range and distribution, which can improve the performance and convergence of the model.
# 
# StandardScaler is a popular scaling technique used in machine learning to standardize features by removing the mean and scaling to unit variance. It is available in the scikit-learn library, which provides a wide range of machine learning tools and preprocessing functions.

# In[39]:


from sklearn.preprocessing import StandardScaler


# In[40]:


scaler = StandardScaler()


# In[41]:


x = scaler.fit_transform(x)


# In[42]:


x


# # Splitting Dataset:

# Dataset splitting is an important step in machine learning and data analysis. It involves dividing a dataset into two or more subsets to train and evaluate a model effectively. The most common type of dataset splitting is into training and testing subsets.
# 
# Splitting the dataset into training, testing, and validation sets is crucial for effective model training and evaluation. The training set (70-80%) is used for model training, the testing set (20-30%) for performance assessment, and the validation set (10-20%) for hyperparameter tuning and preventing overfitting. This ensures the model's ability to generalize to new data and enhances its reliability. 

# In[43]:


from sklearn.model_selection import train_test_split


# In[44]:


x_train, x_test, y_train, y_test = train_test_split(x, 
                                                    y, 
                                                    test_size = 0.2)


# In[45]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[46]:


x_test, x_val, y_test, y_val = train_test_split(x_test, 
                                                y_test, 
                                                test_size = 0.5)


# In[47]:


x_test.shape, x_val.shape, y_test.shape, y_val.shape


# In[48]:


import xgboost as xgb


# In[49]:


model = xgb.XGBClassifier(learning_rate = 0.1, 
                          max_depth = 50, 
                          n_estimator = 100)


# In[51]:


model.fit(x_train, y_train)


# In[52]:


model.score(x_train, y_train)


# In[53]:


model.score(x_test, y_test)


# In[54]:


y_pred = model.predict(x_test)


# In[55]:


y_pred


# In[56]:


print(y_test[20]), print(y_pred[20])


# # Errror Analysis :

# Error analysis is a crucial step in the evaluation and improvement of machine learning models. It involves the systematic examination and understanding of the errors made by the model during prediction. The primary goal of error analysis is to identify patterns and sources of mistakes made by the model, which can provide valuable insights into its performance and guide improvements.

# In[60]:


from sklearn.metrics import confusion_matrix, classification_report


# In[61]:


print(classification_report(y_test, y_pred))


# In[62]:


confusion_matrix = confusion_matrix(y_test, y_pred)


# In[63]:


confusion_matrix


# In[65]:


plt.figure(figsize = (6, 4))
sns.heatmap(confusion_matrix, 
            annot = True, 
            cmap = 'inferno')


# # Thanks !
