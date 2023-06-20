#!/usr/bin/env python
# coding: utf-8

# # Classifying Iris Species

# Classifying Iris species is a popular machine learning project that involves building a model to classify different species of Iris flowers based on their petal and sepal measurements. The Iris dataset is commonly used for this task and is available in many machine learning libraries, including scikit-learn.

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


# # Importing Dataset from sklearn.datasets

# In[2]:


from sklearn.datasets import load_iris
iris_dataset = load_iris()


# In[3]:


print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))


# In[4]:


print(iris_dataset['DESCR'][:193] + "\n...")


# In[5]:


print("Target names: {}".format(iris_dataset['target_names']))


# In[6]:


print("First five columns of data:\n{}".format(iris_dataset['data'][:4]))


# # Splitting Dataset

# Splitting a dataset refers to the process of dividing a given dataset into two or more subsets for training and evaluation purposes. The most common type of split is between the training set and the testing (or validation) set. This division allows us to assess the performance of a machine learning model on unseen data and evaluate its generalization capabilities.
# 
# Train-Test Split: This is the most basic type of split, where the dataset is divided into a training set and a testing set. The training set is used to train the machine learning model, while the testing set is used to evaluate its performance. The split is typically done using a fixed ratio, such as 80% for training and 20% for testing.

# In[7]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(iris_dataset["data"],
                                                    iris_dataset["target"],
                                                    random_state = 0)


# In[8]:


x_train.shape, y_train.shape


# In[9]:


iris_dataframe = pd.DataFrame(x_train, columns = iris_dataset.feature_names)


# In[10]:


iris_dataframe.head()


# # Correlation Matrix

# A correlation matrix is a table that shows the pairwise correlations between variables in a dataset. Each cell in the table represents the correlation between two variables, and the strength and direction of the correlation is indicated by the color and magnitude of the cell.
# 
# Correlation matrices are commonly used in data analysis to identify relationships between variables and to help understand the structure of the data. The values in the correlation matrix range from -1 to 1, with -1 indicating a perfect negative correlation, 1 indicating a perfect positive correlation, and 0 indicating no correlation.

# In[11]:


corr_data = iris_dataframe.corr()


# In[12]:


corr_data


# In[13]:


sns.heatmap(corr_data)


# In[14]:


iris_dataframe.mean()


# # KNeighbors Classifier

# The K-Nearest Neighbors (KNN) Classifier is a machine learning algorithm used for both classification and regression tasks. It is a non-parametric algorithm that makes predictions based on the similarity of a new data point to its neighboring data points in the training dataset.

# In[15]:


from sklearn.neighbors import KNeighborsClassifier


# In[16]:


knn = KNeighborsClassifier()


# In[17]:


knn.fit(x_train, y_train)


# In[18]:


y_pred = knn.predict(x_test)


# In[19]:


y_pred


# In[21]:


from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# In[22]:


accuracy_score_knn = accuracy_score(y_pred, y_test)


# In[23]:


accuracy_score_knn


# In[24]:


print(classification_report(y_test, y_pred))


# In[25]:


plt.figure(figsize = (6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), 
            annot = True, 
            cmap = 'RdPu')


# # Random Forest Classifier

# The Random Forest classifier is an ensemble learning method that combines multiple decision trees to make predictions. It operates by constructing a multitude of decision trees during training and outputs the class that is the mode of the classes predicted by the individual trees.

# In[26]:


from sklearn.ensemble import RandomForestClassifier


# In[27]:


random_forest = RandomForestClassifier()


# In[28]:


random_forest.fit(x_train, y_train)


# In[29]:


random_forest.score(x_train, y_train)


# In[30]:


y_pred_random_forest = random_forest.predict(x_test)


# In[31]:


y_pred_random_forest


# In[33]:


accuracy_score_clf = accuracy_score(y_pred_random_forest, y_test)


# In[34]:


accuracy_score_clf


# In[35]:


print(classification_report(y_pred_random_forest, y_test))


# In[36]:


plt.figure(figsize = (6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), 
            annot = True, 
            cmap = 'RdPu')


# In[37]:


random_forest.score(x_train, y_train)


# # Cross Validation Score

# The cross-validation score is a metric used to evaluate the performance of a model by estimating its accuracy on unseen data. It is obtained by dividing the data into multiple subsets or "folds" and iteratively training and evaluating the model on different combinations of training and testing data.

# In[38]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut


# In[39]:


leave_one = LeaveOneOut()


# In[40]:


kfold = KFold()


# In[41]:


cross_val_acc_kfold = cross_val_score(random_forest, x_train, y_train , cv = kfold)


# In[42]:


cross_val_acc_kfold


# In[43]:


np.mean(cross_val_acc_kfold)


# In[44]:


cross_val_accuracy = cross_val_score(knn, x_train, y_train, cv = 9, scoring = "accuracy")


# In[45]:


cross_val_accuracy


# In[46]:


np.mean(cross_val_accuracy)


# In[47]:


cross_val_acc_leave_one = cross_val_score(knn, x_train, y_train, cv = leave_one)


# In[48]:


cross_val_acc_leave_one


# In[50]:


print(confusion_matrix(y_pred, y_test))


# # Confusion Matrix Display

# The confusion matrix is a table that is used to evaluate the performance of a classification model. It provides a summary of the model's predictions by showing the counts of true positive (TP), true negative (TN), false positive (FP), and false negative (FN) for each class.

# In[51]:


from sklearn.metrics import ConfusionMatrixDisplay


# In[52]:


class_labels = ['class1', 'class2', 'class3'] 
confusion_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=class_labels)
disp.plot()

