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
# Scikit-learn: for machine learning algorithms.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# ## Importing Dataset :

# In[3]:


dataframe  = pd.read_csv('dataset.csv')


# In[4]:


dataframe.head()


# In[5]:


dataframe.tail()


# In[6]:


dataframe.info()


# In[8]:


dataframe.select_dtypes(include = 'object').head()


# In[9]:


dataframe.isna().sum()


# In[11]:


dataframe.describe()


# In[12]:


dataframe.columns


# In[13]:


cont_cols = dataframe.select_dtypes(exclude = ['object']).loc[:, :"days_posted_int"].drop('salary_mentioned', axis = 1)


# In[16]:


cont_cols.head()


# In[19]:


def hist(column):
    sns.histplot(cont_cols.iloc[:, column])
    
for i in np.arange(6):
    plt.rcParams["figure.figsize"] = (11,6)
    plt.subplots_adjust(wspace=0.8,hspace=0.5)
    plt.subplot(3,3,i+1)
    plt.suptitle('Continuous values', fontsize=20)
    hist(i)
    plt.ylabel('')


# ### Insights :

# n reviews, there are noticeable outliers that can skew perceptions. Job listings typically seek around 5 years of experience on average. Most job postings are over a month old, possibly due to specific requirements or slower hiring in certain industries or regions. Job seekers should consider posting dates when applying.

# In[21]:


yn_cols = dataframe.select_dtypes(exclude=["object_"]).loc[:,'python':]


# In[22]:


yn_cols.head()


# In[36]:


def plot(column):
    sns.set_context(rc = {'patch.linewidth': 0.5})
    sns.barplot(x=yn_cols.iloc[:,column].value_counts().index,y=yn_cols.iloc[:,column].value_counts(),edgecolor='black')

for i in np.arange(14):
    plt.rcParams["figure.figsize"] = (15, 14)
    plt.subplots_adjust(wspace=0.8,hspace=0.8)
    plt.subplot(5,3,i+1)
    plt.title(f'{yn_cols.columns[i]}')
    plt.suptitle('Skill requirements', fontsize=20)
    plot(i)
    plt.ylabel('')


# ### Insights :

# Roughly half of the available job listings emphasize the need for Python skills, while fewer than half of them specifically mention SQL as a requirement. This indicates a strong demand for Python proficiency in the job market compared to SQL.

# In[28]:


df_locs = dataframe.loc[:,'bangalore':'pune']


# In[30]:


def plot1(column):
    sns.set_context(rc = {'patch.linewidth': 0.5})
    ax = sns.barplot(x=df_locs.iloc[:,column].value_counts().index,y=df_locs.iloc[:,column].value_counts(),edgecolor='black')

for i in np.arange(9):
    plt.rcParams["figure.figsize"] = (15, 14)
    plt.subplots_adjust(wspace=0.8,hspace=0.8)
    plt.subplot(5,3,i+1)
    plt.suptitle('Data Science jobs by state', fontsize=20)
    plt.title(f'{df_locs.columns[i]}')
    plot1(i)
    plt.ylabel('')


# In[32]:


plt.rcParams["figure.figsize"] = (6,6)
plt.pie(dataframe['salary_mentioned'].value_counts(), labels=['No','Yes'],autopct="%5.2f%%",\
        wedgeprops={"edgecolor":"k",'linewidth': 1})
plt.suptitle('Salary Mentioned',fontsize=20)
plt.show()


# In[34]:


plt.rcParams["figure.figsize"] = (6,6)
plt.pie(dataframe['post'].value_counts(), labels=['Na','Senior','Junior'], autopct="%5.2f%%", \
        wedgeprops={"edgecolor":"k",'linewidth': 1})
plt.suptitle('Job Position',fontsize=20)
plt.show()


# In[35]:


plt.pie(dataframe['job_simp'].value_counts(), labels=['Data Scientist','Na','MLE','Manager','Analyst','Data Engineer'],\
        autopct="%5.2f%%", wedgeprops={"edgecolor":"k",'linewidth': 0.1})
plt.suptitle('Job role',fontsize=20)
plt.show()


# ### Insights :

# Surprisingly, the vast majority of job listings do not include salary information, leaving potential candidates in the dark about compensation. This omission could be due to companies preferring to discuss salary during the interview process or simply not disclosing it upfront. Additionally, approximately one-third of the postings are for senior-level positions, suggesting a demand for experienced professionals. Notably, the majority of job listings are in search of data scientists, which aligns with the keyword used for data scraping—highlighting the relevance of this skillset in the job market.

# In[39]:


plt.rcParams["figure.figsize"] = (10,7)
sns.barplot(x=dataframe.company.value_counts()[:30].index, y=dataframe.company.value_counts()[:30], edgecolor='black')
plt.suptitle('Hiring Companies', fontsize=20)
plt.xticks(rotation=270)
plt.ylabel('')
plt.show()


# In[41]:


pd.pivot_table(dataframe, index=['job_simp','post'], values=['bangalore', 'delhi', 'kolkata',
       'mumbai', 'remote', 'gurgaon', 'hyderabad', 'noida', 'pune'],aggfunc='sum').sort_values('bangalore', ascending=False)


# In[43]:


pd.pivot_table(dataframe, index=['job_simp','post'], values=['ratings','avg_experience','days_posted_int','reviews_int'])\
                                                      .sort_values('avg_experience',ascending=False)


# ### Insights :

# 
# In terms of experience requirements, senior manager positions demand the highest average experience level. Senior data scientist roles typically seek candidates with about 7 years of experience, while junior data scientist positions look for around 4 years of experience on average. It's worth noting that all of the job listings are at least 22 days old, suggesting that these opportunities have been available for some time. Interestingly, companies hiring for analyst positions seem to have the most reviews, indicating a potentially competitive or active job market for this role.

# In[47]:


df_salary = dataframe.loc[dataframe['salary_mentioned'] == 1].copy()

plt.rcParams["figure.figsize"] = (10,6)
sns.barplot(x=df_salary['salary'].value_counts().index, y=df_salary['salary'].value_counts())
plt.xticks(rotation=40)
plt.ylabel('')
plt.suptitle('Salary Range',fontsize=20)
plt.show()


# In[48]:


df_salary.salary = df_salary.salary.apply(lambda x: x.replace('50,000','0.5'))
df_salary['avg_salary'] = df_salary.salary.apply(lambda x: (float(x.split()[0].split('-')[0])+float(x.split()[0].split('-')[1]))/2)
pd.pivot_table(df_salary,index=['location','company','job_simp','post'],values=['avg_salary','avg_experience', 'days_posted_int',
                                                                      'ratings','reviews_int'])


# In[49]:


dataframe.to_csv('visualized.csv')

