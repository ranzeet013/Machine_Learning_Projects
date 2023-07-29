#!/usr/bin/env python
# coding: utf-8

# # Predicting Student Success Rate 

# Predicting student success rate is a data science project that aims to develop a machine learning model capable of forecasting students' academic performance. The project involves using historical student data, such as demographics, previous qualifications, attendance, and curricular performance, as features to predict the target variable, which is typically a measure of academic success (e.g., final grades, graduation status, or GPA).

# # Importing Libraries :

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


# # Importing Dataset :

# The dataset provides comprehensive information about students' enrollment in various courses and includes data related to their personal characteristics, academic performance, and economic indicators. It encompasses fields such as marital status, application mode, and age at enrollment, shedding light on the diversity of students and their preferred application methods. Additionally, the dataset delves into the students' educational backgrounds, capturing details about their previous qualifications and the qualifications of their parents, which could potentially influence their academic journey. Furthermore, the dataset contains data on the students' attendance preferences, nationality, and special needs, allowing for a more nuanced understanding of the student population. Moreover, it includes valuable information on the curricular units in the first and second semesters, covering credit details, evaluations, approvals, and grades. Importantly, the dataset also incorporates economic indicators such as the unemployment rate, inflation rate, and GDP, which may offer insights into any correlations between economic conditions and academic outcomes. Lastly, the 'Target' column provides a focal point for analysis, indicating a variable of interest that researchers or analysts may seek to predict or understand better. Overall, this dataset presents a rich opportunity for exploring the complex interplay between student characteristics, academic performance, and economic factors

# In[3]:


dataframe = pd.read_csv('dataset.csv')


# # Exploratory Data Analysis :

# Exploratory Data Analysis (EDA) is an essential process in data analysis that involves visually and statistically exploring a dataset to gain insights into its structure and relationships. It includes examining data distributions, correlations, and dependencies among variables. EDA helps identify outliers, missing values, and trends, leading to data preprocessing decisions and informing subsequent analysis steps. The process employs various data visualization techniques and is iterative, allowing for deeper understanding and generation of new research questions. Overall, EDA provides a solid foundation for meaningful data interpretation and effective decision-making.

# In[4]:


dataframe.head()


# In[5]:


dataframe.tail()


# In[6]:


dataframe.info()


# In[7]:


dataframe.isna().sum()


# # Statical Info

# Exploratory Data Analysis (EDA) involves a comprehensive examination of the dataset's statistical information to gain meaningful insights into its underlying patterns and characteristics. During this process, various key statistical measures are calculated and analyzed. Summary statistics, such as the mean, median, mode, standard deviation, minimum, maximum, and quartiles, offer an overview of the central tendencies and dispersion of numerical data. For categorical variables, count and frequency are determined to understand the distribution of each category. Correlation analysis reveals the strength and direction of linear relationships between numerical variables, while histograms and bar charts visually depict the distribution of data and frequency of categories, respectively. 

# In[8]:


dataframe.describe()


# In[9]:


dataframe.columns


# In[10]:


dataframe = dataframe[dataframe .Target!='Enrolled']


# In[11]:


dataframe.columns


# # Encoding :

# 
# Encoding refers to the process of converting data from one format or representation to another. In the context of data analysis and machine learning, encoding is often used to transform categorical or text data into numerical form, which can be more easily processed and utilized by algorithms.

# Label Encoding: In label encoding, each unique category in a categorical variable is assigned an integer label. For example, if we have categories like "Red," "Green," and "Blue," they could be encoded as 0, 1, and 2, respectively. However, caution should be exercised when using label encoding for ordinal data, as the numerical representation may introduce an unintended ordinal relationship.

# In[12]:


dataframe['Target'].value_counts()


# In[13]:


from sklearn.preprocessing import LabelEncoder


# In[14]:


encoder = LabelEncoder()


# In[15]:


dataframe['Target'] = encoder.fit_transform(dataframe['Target'])


# In[16]:


dataframe['Target'].value_counts()


# # Distribution Table of Graduate and Non Graduate 

# In[17]:


dataframe['Target'].value_counts().plot(kind = 'bar', 
                                        title = 'Distrbution of Target Value', 
                                        cmap = 'plasma')


# # Correlation Matrix :

# A correlation matrix is a table that shows the pairwise correlations between variables in a dataset. Each cell in the table represents the correlation between two variables, and the strength and direction of the correlation is indicated by the color and magnitude of the cell.
# 
# Correlation matrices are commonly used in data analysis to identify relationships between variables and to help understand the structure of the data. The values in the correlation matrix range from -1 to 1, with -1 indicating a perfect negative correlation, 1 indicating a perfect positive correlation, and 0 indicating no correlation.

# In[18]:


corr_matrix = dataframe.corr()


# In[19]:


corr_matrix


# In[20]:


plt.figure(figsize = (25, 25))
sns.heatmap(corr_matrix, 
            annot = True, 
            cmap = 'inferno')


# # Feature Selection :

# Feature selection is an essential process in data analysis and machine learning, focused on choosing the most relevant and informative subset of features from the original dataset. The primary goal is to improve the model's performance, reduce complexity, and enhance interpretability. By eliminating irrelevant, redundant, or noisy features, feature selection helps prevent overfitting and enables the model to generalize better to new data. This, in turn, reduces computation time and memory requirements during model training and evaluation. Various techniques can be employed for feature selection, including filter methods that rank features based on statistical measures, wrapper methods that utilize the model's predictive performance, and embedded methods that incorporate feature selection within the model training process. Additionally, dimensionality reduction techniques like PCA offer an alternative approach by projecting the data onto a lower-dimensional space. Ultimately, selecting the right features is critical to producing more efficient and accurate models, facilitating data-driven insights and decision-making with enhanced clarity and efficiency.
# 
# 
# 
# 
# 
# 

# In[21]:


dataframe.columns


# In[22]:


dataframe.drop(['International',
                'Nacionality', 
                "Father's qualification",
                'Curricular units 1st sem (credited)',
                'Curricular units 1st sem (enrolled)',
                'Curricular units 1st sem (approved)','Course',
                'Educational special needs','Unemployment rate',
                'Inflation rate'],axis=1,inplace=True)


# In[23]:


dataframe.head()


# In[24]:


dataframe.columns


# In[25]:


corr_matrix = dataframe.corr()


# In[26]:


corr_matrix


# In[27]:


plt.figure(figsize = (25, 25))
sns.heatmap(corr_matrix, 
            annot = True, 
            cmap = 'inferno')


# In[28]:


dataframe.columns


# # Correlation Of Graduation Rate :

# In[29]:


dataset = dataframe.drop('Target', axis = 1)


# In[30]:


dataset.head()


# In[31]:


dataset.corrwith(dataframe['Target']).plot.bar(
    title = 'Correlation With Dropout', 
    figsize = (12, 5) , 
    cmap = 'plasma'
)


# In[32]:


dataframe.head()


# # Splitting Dataset :

# Splitting a dataset refers to the process of dividing a given dataset into two or more subsets for training and evaluation purposes. The most common type of split is between the training set and the testing (or validation) set. This division allows us to assess the performance of a machine learning model on unseen data and evaluate its generalization capabilities.
# 
# Train-Test Split: This is the most basic type of split, where the dataset is divided into a training set and a testing set. The training set is used to train the machine learning model, while the testing set is used to evaluate its performance. The split is typically done using a fixed ratio, such as 80% for training and 20% for testing.

# In[33]:


x = dataframe.drop('Target', axis = 1)
y = dataframe['Target']


# In[34]:


x.shape, y.shape


# In[35]:


from sklearn.model_selection import train_test_split


# In[36]:


x_train, x_test, y_train, y_test = train_test_split(x,  
                                                    y, 
                                                    test_size = 0.3, 
                                                    stratify = y, 
                                                    random_state = 42)


# In[37]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# # Scaling :

# Scaling is a preprocessing technique used in machine learning to transform the input features to a similar scale. It is often necessary because features can have different units, ranges, or magnitudes, which can affect the performance of certain algorithms. Scaling ensures that all features contribute equally to the learning process and prevents features with larger values from dominating those with smaller values.
# 
# StandardScaler is a commonly used method for scaling numerical features in machine learning. It is part of the preprocessing module in scikit-learn, a popular machine learning library in Python.

# In[38]:


from sklearn.preprocessing import StandardScaler


# In[39]:


scaler = StandardScaler()


# In[40]:


x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# In[41]:


x_train


# In[42]:


x_test


# # Model Selection And Training :

# Model Selection:
# 
# Model selection involves choosing the best algorithm or model architecture for the given problem and dataset. This step requires careful consideration of various factors, such as the nature of the data (e.g., numerical or categorical), the problem type (e.g., regression, classification, clustering), the amount of available data, and the desired model performance. It is essential to select a model that can effectively capture the underlying patterns in the data and make accurate predictions.

# Model Training:
# 
# Once the appropriate model has been selected, the next step is to train it on the dataset. Model training involves adjusting the model's parameters using the training data to make accurate predictions on unseen data. The goal is to minimize the difference between the model's predictions and the actual target values during training.

# In[43]:


x_train.shape, x_test.shape


# In[44]:


from sklearn.ensemble import RandomForestClassifier


# In[45]:


clf = RandomForestClassifier()


# In[46]:


clf.fit(x_train, y_train)


# In[47]:


y_pred = clf.predict(x_test)


# In[48]:


y_pred


# In[49]:


clf.score(x_test, y_test)


# # Hyper-parameter Tuning :

# Hyperparameter tuning is a critical process in machine learning that involves finding the optimal set of hyperparameters for a given model. Hyperparameters are configuration settings that are not learned from the data during model training but are set before the training process begins. They significantly impact the model's performance and generalization ability.
# 
# The goal of hyperparameter tuning is to systematically search through different combinations of hyperparameters to identify the configuration that yields the best model performance. The process ensures that the model is well-optimized and capable of making accurate predictions on new, unseen data.

# Grid Search: 
# 
# Grid search involves specifying a list of values for each hyperparameter. The algorithm then exhaustively tries all possible combinations of hyperparameters to find the best one. This method is simple and can be effective for a small number of hyperparameters, but it becomes computationally expensive as the number of hyperparameters increases.

# In[50]:


from sklearn.model_selection import GridSearchCV


# In[51]:


param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None , 10, 20],
    'min_samples_split': [2,4,5],
    'min_samples_leaf': [1,2,4],
}


# In[52]:


clf_grid=GridSearchCV(estimator=clf, 
                     param_grid=param_grid,
                     cv=3,
                     verbose=0,
                     n_jobs=-1,
                     return_train_score=False)


# In[53]:


clf_grid.fit(x_train,y_train)


# In[54]:


clf_grid.best_params_


# In[55]:


random_forest = RandomForestClassifier(**clf_grid.best_params_)


# In[56]:


random_forest.fit(x_train, y_train)


# In[57]:


y_pred = random_forest.predict(x_test)


# In[58]:


y_pred


# In[59]:


print(y_test.iloc[20]), print(y_pred[20])


# In[60]:


print(y_test.iloc[30]), print(y_pred[30])


# # Error Analysis :

# Error analysis is a crucial process in machine learning and data analysis, aiming to understand model errors and patterns of misclassifications. By investigating causes and impacts of errors, analysts can devise strategies to improve the model's performance, leading to more accurate and reliable predictions. This iterative process enables continuous refinement, making machine learning systems more effective in real-world applications.

# In[61]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score


# In[62]:


accuracy_score = accuracy_score(y_test, y_pred)
print('Accuracy_score: ', accuracy_score)


# In[63]:


precision_score = precision_score(y_test, y_pred)
print('Precision_score: ', precision_score)


# In[64]:


f1_score = f1_score(y_test, y_pred)
print('f1_score: ', f1_score)


# In[65]:


print(classification_report(y_test, y_pred))


# In[66]:


confusion_matrix = confusion_matrix(y_test, y_pred)


# In[67]:


confusion_matrix


# In[68]:


plt.figure(figsize = (6, 4))
sns.heatmap(confusion_matrix, 
            annot = True, 
            cmap = 'RdPu')


# # Thanks !
