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

# In[2]:


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

# In[3]:


dataframe = pd.read_csv('911.csv')


# In[4]:


dataframe.head()


# In[5]:


dataframe.tail()


# In[6]:


dataframe.shape


# ## Investigating Outliners :

# In[31]:


dataframe['Date'] = pd.to_datetime(dataframe['Date'])


# In[33]:


dataframe.groupby(dataframe[dataframe['Date'].dt.year>=2018]['Date']).count().plot.line(use_index = True, y = 'title', legend = None)
plt.ylabel('count')


# In[35]:


dataframe.groupby(dataframe[(dataframe['Date'].dt.year>= 2018) & (dataframe['Date'].dt.month==3)]['Date']).count()


# In[36]:


dataframe[dataframe['Date']=='2018-03-02']['Reason'].value_counts()


# In[37]:


sns.countplot(x='Reason',data=dataframe[dataframe['Date']=='2018-03-02'])


# In[41]:


dataframe.groupby(dataframe[(dataframe['Date'].dt.year>= 2018) & (dataframe['Date'].dt.month==11)]['Date']).count()


# In[42]:


sns.countplot(x = 'Reason',data = dataframe[dataframe['Date']=='2018-11-15'])


# In[43]:


dataframe.groupby(['Date','Reason']).count().unstack()


# In[44]:


dataframe.groupby(['Date','Reason']).count()['title'].unstack().plot.line(use_index = True, 
                                                                          y = 'Traffic', 
                                                                          figsize= (15,2), 
                                                                          legend = None)
plt.title('Traffic')
plt.ylabel('count')


# In[45]:


dataframe.groupby(['Date','Reason']).count()['title'].unstack().plot.line(use_index = True, 
                                                                          y = 'EMS', 
                                                                          figsize= (15,2), 
                                                                          legend = None)
plt.title('EMS')
plt.ylabel('count')


# In[46]:


dataframe.groupby(['Date','Reason']).count()['title'].unstack().plot.line(use_index = True, 
                                                                          y = 'Fire', 
                                                                          figsize= (15,2), 
                                                                          legend = None)
plt.title('Fire')
plt.ylabel('count')


# In[ ]:




