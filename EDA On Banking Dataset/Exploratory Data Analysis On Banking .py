#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries :

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

import warnings
warnings.filterwarnings('ignore')


# ## Importing Dataset :

# 
# The dataset appears to be related to marketing or customer interactions and related to a bank or financial institution. Here's a brief description of each column:

# 
# 
# 1. `age`: Represents the age of the individual.
# 2. `job`: Describes the occupation or job of the person.
# 3. `marital`: Indicates the marital status of the person (e.g., married, single, divorced).
# 4. `education`: Represents the educational level of the person (e.g., basic, high school, university).
# 5. `default`: Indicates whether the person has credit in default ('yes', 'no', or 'unknown').
# 6. `housing`: Shows whether the person has a housing loan ('yes', 'no', or 'unknown').
# 7. `loan`: Indicates whether the person has a personal loan ('yes', 'no', or 'unknown').
# 8. `contact`: Describes the method of communication used to contact the person (e.g., 'cellular', 'telephone').
# 9. `month`: Represents the month of the last contact.
# 10. `day_of_week`: Indicates the day of the week of the last contact.
# 11. `duration`: Represents the duration of the last contact in seconds.
# 12. `campaign`: Indicates the number of contacts made during this campaign.
# 13. `pdays`: Describes the number of days since the person was last contacted or -1 if they were not previously contacted.
# 14. `previous`: Represents the number of contacts made before this campaign.
# 15. `poutcome`: Indicates the outcome of the previous marketing campaign.
# 16. `emp.var.rate`: Describes the employment variation rate.
# 17. `cons.price.idx`: Represents the consumer price index.
# 18. `cons.conf.idx`: Indicates the consumer confidence index.
# 19. `euribor3m`: Represents the Euribor 3-month rate.
# 20. `nr.employed`: Describes the number of employees.
# 21. `y`: The target variable, indicating whether the person subscribed to a term deposit ('yes' or 'no').
# 
# 

# In[5]:


dataframe = pd.read_csv('bank-additional.csv', sep = ';')


# In[6]:


dataframe.head()


# In[7]:


dataframe.tail()


# In[8]:


dataframe.shape


# In[9]:


dataframe.info()


# In[10]:


dataframe.select_dtypes(include= 'object').head()


# In[14]:


dataframe.describe(include = 'object')


# The result shows that the average client refers to administrative staff (job = admin.), is married (marital = married) and has a university degree (education = university.degree).

# In[12]:


dataframe.isna().sum()


# Method describe shows the main statistical characteristics of the dataset for each numerical feature (int64 and float64 types): the existing values number, mean, standard deviation, range, min & max, 0.25, 0.5 and 0.75 quartiles.

# In[13]:


dataframe.describe()


# In[16]:


dataframe.columns


# In[19]:


dataframe["y"].value_counts()


# 4640 clients (11.3%) of 41188 issued a term deposit, the value of the variable y equals yes.

# In[26]:


dataframe['y'].value_counts().plot(kind = 'bar', 
                                 figsize = (12, 5), 
                                 title = 'Distribution', 
                                 cmap = 'ocean')


# Let's look at the client distribution by the variable marital. Specify the value of the normalize = True parameter to view relative frequencies, but not absolute.

# In[20]:


dataframe["marital"].value_counts(normalize = True)


# As we can see, 61% (0.61) of clients are married, which must be taken into account when planning marketing campaigns to manage deposit operations.

# In[27]:


dataframe['marital'].value_counts(normalize = True).plot(kind = 'bar', 
                                                         figsize = (12, 5), 
                                                         title = 'Distribution', 
                                                         cmap = 'ocean')


# A DataFrame can be sorted by a few feature values. In our case, for example, by duration (ascending = False for sorting in descending order):

# In[28]:


dataframe.sort_values(by = "duration", ascending = False).head()


# The sorting results show that the longest calls exceed one hour, as the value duration is more than 3600 seconds or 1 hour. At the same time, it usually was on Mondays and Thursdays (day_of_week) and, especially, in November and August (month).
# 
# Sort by the column group:

# In[29]:


dataframe.sort_values(by = ["age", "duration"], ascending = [True, False]).head()


# In[30]:


corr_matrix = dataframe.corr()


# In[31]:


corr_matrix


# In[33]:


plt.figure(figsize = (12, 5))
sns.heatmap(corr_matrix, 
            annot = True, 
            cmap = 'coolwarm')


# In[34]:


dataframe.head()


# In[40]:


pd.crosstab(dataframe["y"], dataframe["marital"])


# The result shows that the number of attracted married clients is 2532 (y = 1 for married) from the total number.

# In[41]:


pd.crosstab(dataframe["y"],
            dataframe["marital"],
            normalize = 'index')


# We see that more than half of the clients (61%, column married) are married and have not issued a deposit.
# 
# In Pandas, pivot tables are implemented by the method pivot_table with such parameters:
# 
# values – a list of variables to calculate the necessary statistics,
# index – a list of variables to group data,
# aggfunc — values that we actually need to count by groups - the amount, average, maximum, minimum or something else.

# In[42]:


dataframe.pivot_table(
    ["age", "duration"],
    ["job"],
    aggfunc = "mean",
).head(10)


# Method scatter_matrix allows you to visualize the pairwise dependencies between the features (as well as the distribution of each feature on the diagonal). We will do it for numerical features.

# In[44]:


pd.plotting.scatter_matrix(
    dataframe[["age", "duration", "campaign"]],
    figsize = (15, 15),
    diagonal = "kde", 
    cmap = 'coolwarm')
plt.show()


# A scatter matrix (pairs plot) compactly plots all the numeric variables we have in a dataset against each other. The plots on the main diagonal allow you to visually define the type of data distribution: the distribution is similar to normal for age, and for a call duration and the number of contacts, the geometric distribution is more suitable.

# In[45]:


dataframe["age"].hist()


# In[46]:


dataframe.hist(color = "k",
        bins = 30,
        figsize = (15, 10))
plt.show()


# A visual analysis of the histograms presented allows us to make preliminary assumptions about the variability of the source data.
# 
# Now we will use Box Plot. It will allow us to compactly visualize the main characteristics of the feature distribution (the median, lower and upper quartile, minimal and maximum, outliers).

# In[47]:


dataframe.boxplot(column = "age",
           by = "marital")
plt.show()


# The plot shows that unmarried people are on average younger than divorced and married ones. For the last two groups, there is an outlier zone over 70 years old, and for unmarried - over 50.

# In[48]:


dataframe.boxplot(column = "age",
           by = ["marital", "housing"],
           figsize = (20, 20))
plt.show()


# In[50]:


dataframe.sort_values(by = "campaign", ascending = False).head(10)


# Determine the median age and the number of contacts for different levels of client education.

# In[52]:


dataframe.pivot_table(
    ["age", "campaign"],
    ["education"],
    aggfunc = ["mean", "count"],
)


# Output box plot to analyze the client age distribution by their education level.

# In[53]:


dataframe.boxplot(column = "age",
  by = "education",
  figsize = (15, 15))
plt.show()


# In[ ]:




