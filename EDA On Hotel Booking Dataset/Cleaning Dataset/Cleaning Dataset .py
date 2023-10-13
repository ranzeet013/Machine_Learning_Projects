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

import warnings
warnings.filterwarnings('ignore')


# ### Importing Dataset :

# The dataset consists of the following attributes :
# 1. 'hotel': Type of hotel (e.g., resort hotel or city hotel).
# 2. 'is_canceled': A binary indicator (1 or 0) to show whether the booking was canceled or not.
# 3. 'lead_time': The number of days between booking and arrival.
# 4. 'arrival_date_year': The year of arrival.
# 5. 'arrival_date_month': The month of arrival.
# 6. 'arrival_date_week_number': The week number of arrival.
# 7. 'arrival_date_day_of_month': The day of the month of arrival.
# 8. 'stays_in_weekend_nights': The number of weekend nights (Saturday or Sunday) the guest stayed.
# 9. 'stays_in_week_nights': The number of weekday nights (Monday to Friday) the guest stayed.
# 10. 'adults': The number of adults in the booking.
# 11. 'children': The number of children in the booking.
# 12. 'babies': The number of babies in the booking.
# 13. 'meal': The type of meal included in the booking.
# 14. 'country': The country of origin of the guest.
# 15. 'market_segment': The market segment of the booking (e.g., online travel agencies, direct, corporate).
# 16. 'distribution_channel': The distribution channel used for the booking.
# 17. 'is_repeated_guest': A binary indicator for whether the guest is a repeated visitor (1 or 0).
# 18. 'previous_cancellations': The number of previous booking cancellations by the guest.
# 19. 'previous_bookings_not_canceled': The number of previous bookings not canceled by the guest.
# 20. 'reserved_room_type': The room type reserved by the guest.
# 21. 'assigned_room_type': The room type assigned to the guest upon arrival.
# 22. 'booking_changes': The number of changes made to the booking.
# 23. 'deposit_type': The type of deposit made for the booking.
# 24. 'agent': The ID of the travel agency or booking agent.
# 25. 'company': The ID of the company that made the booking.
# 26. 'days_in_waiting_list': The number of days the booking was on a waiting list.
# 27. 'customer_type': The type of customer (e.g., transient, contract, group).
# 28. 'adr': The average daily rate (price) for the booking.
# 29. 'required_car_parking_spaces': The number of car parking spaces required by the guest.
# 30. 'total_of_special_requests': The total number of special requests made by the guest.
# 31. 'reservation_status': The reservation status (e.g., checked-in, canceled).
# 32. 'reservation_status_date': The date when the reservation status was last updated.
# 

# In[3]:


dataframe = pd.read_csv('Hotel Bookings.csv')


# In[4]:


dataframe.head()


# In[5]:


dataframe.tail()


# In[7]:


dataframe.head().T


# In[8]:


dataframe.shape


# In[9]:


dataframe.info()


# In[10]:


dataframe.isnull().sum().sort_values(ascending = True)


# As we can see, we have a lot of Null values in columns 'company' and 'agent,' which significantly impact our further analysis of the data. As a result, we have decided to remove these two columns for our subsequent analysis.

# In[11]:


dataframe.columns


# In[12]:


dataframe.describe()


# ### Cleaning Dataset :

# We've decided to remove some columns from our dataset because they don't appear to be important for our study. These columns include 'arrival_date_week_number,' 'arrival_date_day_of_month,' 'company,' 'previous_cancellations,' 'previous_bookings_not_canceled,' and 'agent.' By dropping these columns, we're simplifying our data to focus on the information that matters most for our analysis, making it easier to work with and drawing our attention to the key factors in our study.

# In[16]:


dataframe.drop(['arrival_date_week_number','arrival_date_day_of_month','company','agent','previous_cancellations','previous_bookings_not_canceled','reservation_status_date'], axis=1, inplace=True)


# In[17]:


dataframe.head().T


# In[18]:


dataframe.isna().sum()


# If there are only a few missing numerical values in a column and it won't greatly affect your analysis, you can replace those missing values with the median value of the column. This helps keep your data complete and allows you to continue your analysis without disruptions caused by missing information.

# In[20]:


dataframe['children'].fillna(dataframe['children'].median(), inplace = True)


# In[21]:


dataframe.isna().sum()


# Further cleaning the data, when the percentage of null values in the 'country' column is less than 0.5%, it's a reasonable approach to impute (fill in) those null values with the mode. The mode represents the most frequently occurring value in the 'country' column and can serve as a good estimate for the missing data. This helps ensure that only a minimal portion of your data is missing and doesn't significantly impact your analysis, while still maintaining the integrity of the dataset.

# In[22]:


dataframe.loc[dataframe['country'] == 'PRT', 'country'].count()


# In[25]:


dataframe['country'].fillna(dataframe['country'].mode()[0], inplace = True)


# In[26]:


dataframe.isna().sum()


# In[29]:


dataframe.to_csv('cleaned_dataframe')


# In[ ]:




