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

# In[2]:


dataframe = pd.read_csv('911.csv')


# In[3]:


dataframe.head()


# ## Heatmap : 

# A heatmap is a data visualization technique that uses color to represent the magnitude of values in a matrix. It is important because it provides a quick and intuitive way to understand complex data patterns, relationships, and variations, making it useful in fields such as data analysis, biology, and finance for identifying trends and anomalies.

# In[17]:


dataframe_heatmap = dataframe.groupby(['Day of Week','Hour']).count().unstack()['title']


# In[18]:


dataframe_heatmap


# In[19]:


fig, ax = plt.subplots(figsize=(12,6))
sns.heatmap(dataframe_heatmap, 
            cmap='coolwarm', 
            ax = ax)


# When we look at the data, we notice that most of the phone calls happen during the daytime, which is pretty much what we would expect. Also, these calls are most common on regular business days. To get a better understanding of these patterns, we can use a cluster map. Think of it like grouping similar things together on a map to see if there are any specific trends or commonalities. It helps us see if certain days or times are more alike in terms of call density, which can give us more insights into the data.

# In[20]:


sns.clustermap(dataframe_heatmap, 
               cmap = 'coolwarm', 
               figsize = (12,10))


# The cluster map provides a clearer picture, highlighting that the highest call density occurs on weekdays, particularly during the standard working hours from 9 am to 6 pm. It visually confirms that these specific times and days of the week stand out as the busiest periods for phone calls.

# In[21]:


dataframe_heatmap = dataframe.groupby(['Day of Week','Month']).count().unstack()['title']
dataframe_heatmap


# In[22]:


fig, ax = plt.subplots(figsize=(12,6))
sns.heatmap(dataframe_heatmap, 
            cmap='coolwarm', 
            ax = ax)


# The peak density in phone calls falls on a specific day: Friday in March. This spike can be attributed to the weather incident we examined earlier, which occurred on Friday, March 2, 2018. However, it's crucial to remember that our dataset only goes up until mid-November. Therefore, we can't rely on data for November and December to draw significant conclusions because of the limited amount of information available during those months.

# In[23]:


sns.clustermap(dataframe_heatmap, 
               cmap = 'coolwarm', 
               figsize = (12,10))


# The clustermap makes it abundantly clear that Sundays consistently have the lowest number of 911 calls when compared to the other days of the week.

# ## Conclusion :

# In my visual analysis, I had the opportunity to practice a variety of data visualization techniques as I delved into this dataset. I used pandas to create dataframes, enabling me to manipulate, extract, and visualize important data categories of interest. As I explored the dataset, I discovered that EMS-related calls accounted for the highest proportion of 911 calls, followed by traffic-related calls and then fire-related calls. I also identified two distinct outliers, one on March 2nd, 2018, and the other on November 15th, 2018, which both appeared to be linked to severe weather conditions. It's worth noting that with further investigation, one could potentially extract valuable insights from the data associated with these specific dates.

# In[ ]:




