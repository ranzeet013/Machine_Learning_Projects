#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sportsdataverse.nba import load_nba_player_boxscore

pd.set_option("display.max_columns", 75)


# In[2]:


dataframe = load_nba_player_boxscore(seasons=2022)\
    .to_pandas(date_as_object=False)


# In[3]:


dataframe.head()


# In[4]:


dataframe.tail()


# In[5]:


dataframe.info()


# In[6]:


dataframe.describe()


# In[7]:


dataframe.isna().sum()


# In[9]:


stephen = dataframe.query("athlete_display_name == 'Stephen Curry'")


# In[10]:


stephen.head()


# In[11]:


stephen.info()


# In[12]:


stephen.to_csv('stephen.csv', index = False)


# In[13]:


stephen = pd.read_csv('stephen.csv')


# In[14]:


stephen.head()


# In[15]:


stephen.tail()


# In[17]:


stephen.describe()


# In[18]:


stephen['minutes'].max()


# In[19]:


stephen.query("minutes >= 40")


# In[21]:


stephen.query("minutes >= 42 & free_throws_attempted < 10")


# In[22]:


stephen.query("minutes == minutes.min()")


# In[23]:


stephen.points.value_counts().sort_index()


# ### Distribution of Lebron's points :

# In[24]:


stephen.points.value_counts().sort_index().plot();


# In[ ]:




