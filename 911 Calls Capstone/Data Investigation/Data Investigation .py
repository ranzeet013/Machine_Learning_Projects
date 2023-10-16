#!/usr/bin/env python
# coding: utf-8

# ## Importing Dataset :

# These are just a few examples of popular Python libraries. You can import any other library using the same import statement followed by the library name or alias:
# 
# NumPy: for numerical operations and array manipulation
# 
# Pandas: for data manipulation and analysis
# 
# Matplotlib: for creating visualizations

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# ## Importing Dataframe :

# The dataset consists of several columns, each providing different pieces of information related to emergency 911 calls. Here's a brief description of each of the columns:
# 
# 1. 'lat' - This column likely contains the latitude coordinates of the location where the emergency call was made.
# 
# 2. 'lng' - This column is expected to hold the longitude coordinates of the emergency call location.
# 
# 3. 'desc' - This column should contain a description of the incident or emergency that prompted the 911 call. It provides details about what happened.
# 
# 4. 'zip' - This column is likely to store the ZIP code associated with the location of the emergency.
# 
# 5. 'title' - The 'title' column probably holds information about the type or category of the emergency or incident, such as "Fire," "Traffic Accident," or "Medical Emergency."
# 
# 6. 'timeStamp' - This column should include the timestamp indicating when the 911 call was made or received.
# 
# 7. 'twp' - 'twp' likely stands for "township" and contains the name or identifier of the township or area within Montgomery County where the emergency occurred.
# 
# 8. 'addr' - The 'addr' column likely provides the specific address or location details of the emergency.
# 
# 9. 'e' - This column might be an identifier or code associated with the emergency event.
# 
# 10. 'Reason' - The 'Reason' column could be a derived feature that categorizes the reason for the emergency call based on the information in the 'title' column. For example, it might categorize calls as "Medical," "Fire," or "Traffic" based on the incident type.
# 

# In[2]:


dataframe = pd.read_csv('911.csv')


# In[3]:


dataframe.head()


# In[4]:


dataframe.tail()


# In[5]:


dataframe.shape


# In[6]:


dataframe.info()


# In[7]:


dataframe.select_dtypes(include = 'object').head()


# In[8]:


dataframe.isna().sum()


# Keep in mind that a substantial portion of the zip code and some township data is missing in the dataset. So, when analyzing the top values for zip codes and townships, remember that the results may not reflect the complete dataset due to missing information.

# In[9]:


dataframe['zip'].value_counts().head()


# In[11]:


dataframe['twp'].value_counts().head()


# In[12]:


dataframe['Reason'] = dataframe['title'].apply(lambda st: st.split(':')[0])


# In[13]:


dataframe.head()


# In[14]:


dataframe['Reason'].value_counts().head()


# In[15]:


sns.countplot(x = 'Reason', data = dataframe)


# From the graph you've shown, it's clear that fire-related incidents are less represented in your dataset compared to EMS (Emergency Medical Services) and traffic-related incidents. This observation highlights the relatively low frequency of fire-related emergency calls in your dataset, which can be an essential insight when considering the distribution of different types of emergencies.

# ### Exploring Time Data :

# In[16]:


type(dataframe['timeStamp'].iloc[0])


# In[17]:


dataframe.columns


# In[18]:


dataframe['timeStamp'] = pd.to_datetime(dataframe['timeStamp'])


# In[19]:


dataframe['Hour'] = dataframe['timeStamp'].apply(lambda time: time.hour)


# In[20]:


dataframe['Month'] = dataframe['timeStamp'].apply(lambda time: time.month)


# Using the `time.weekday_name` attribute might have been a more direct way to achieve the same solution, but your decision to practice mapping a dictionary is a valuable learning experience. Mapping a dictionary can be a useful skill, and it allows you to have more control and flexibility in your data analysis and visualization. It's a great way to expand your understanding of Python and data manipulation techniques.

# In[21]:


dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
dataframe['Day of Week'] = dataframe['timeStamp'].apply(lambda time: time.dayofweek).map(dmap)


# In[22]:


dataframe.head()


# In[23]:


sns.countplot(x='Day of Week',
              data = dataframe, 
              hue = 'Reason', 
              palette = 'Set2')
plt.legend(loc='lower left',bbox_to_anchor=(1.0,0.5))


# In[24]:


sns.countplot(x='Month', 
              data = dataframe, 
              hue = 'Reason', 
              palette = 'Set2')
plt.legend(loc='lower left',bbox_to_anchor=(1.0,0.5))


# At first glance, it's evident that traffic-related emergency calls tend to decrease on weekends, and the number of fire-related calls is significantly lower per month compared to EMS (Emergency Medical Services) and traffic-related calls. This observation suggests distinct patterns in the data based on the nature of the emergency, with traffic calls showing a weekend decline and fire calls being comparatively infrequent.

# In[25]:


dataframe.groupby('Month').count()


# Creating a graph to visualize the calling trends per month is an excellent way to gain a clearer understanding of the data. By doing so, you can easily identify patterns and trends in emergency call volume over different months, making it more accessible for interpretation and analysis.

# In[26]:


dataframe.groupby('Month').count().plot.line(use_index = True,y = 'title',legend = None)
plt.ylabel('count')


# In[27]:


sns.lmplot(x='Month', 
           y = 'title', 
           data = dataframe.groupby('Month').count().reset_index())
plt.ylabel('count')


# To better understand the data and identify the general trend, especially amidst the numerous spikes in the graph, it's a good idea to perform a linear regression analysis. This will help you establish a trendline that summarizes the overall pattern in the data, making it easier to draw insights and conclusions from your analysis.

# In[29]:


dataframe['Date'] = dataframe['timeStamp'].apply(lambda ts: ts.date())


# In[30]:


dataframe.head()


# In[31]:


dataframe.groupby('Date').count().plot.line(use_index = True, y = 'title', figsize= (15,2), legend = None)
plt.ylabel('count')


# We notice giant outliers in March of 2018 and in November of 2018.

# In[ ]:




