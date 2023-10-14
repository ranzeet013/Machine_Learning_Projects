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
import seaborn as sns
import matplotlib.pyplot as plt 

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# ### Importing Cleaned Dataframe :

# DataFrame with the following columns:
# 
# 1. 'hotel': This column likely contains information about the type of hotel, such as "Resort Hotel" or "City Hotel."
# 
# 2. 'is_canceled': This column probably indicates whether a reservation was canceled or not. It might have binary values like 0 for not canceled and 1 for canceled.
# 
# 3. 'lead_time': This column likely represents the number of days between booking and the arrival date.
# 
# 4. 'arrival_date_year': This column specifies the year of the arrival date.
# 
# 5. 'arrival_date_month': This column contains the month of the arrival date, e.g., "January," "February," etc.
# 
# 6. 'stays_in_weekend_nights': This column probably records the number of weekend nights the guests planned to stay.
# 
# 7. 'stays_in_week_nights': This column probably records the number of weekday nights the guests planned to stay.
# 
# 8. 'adults': This column likely represents the number of adults in the reservation.
# 
# 9. 'children': This column may indicate the number of children in the reservation.
# 
# 10. 'babies': This column may indicate the number of infants (babies) in the reservation.
# 
# 11. 'meal': This column likely specifies the meal package or type, such as "BB" (Bed & Breakfast) or "HB" (Half Board).
# 
# 12. 'country': This column probably indicates the country of origin of the guests.
# 
# 13. 'market_segment': This column may describe the market segment to which the reservation belongs, such as "Online Travel Agents" or "Corporate."
# 
# 14. 'distribution_channel': This column likely describes the distribution channel through which the reservation was made, e.g., "Direct" or "Travel Agents."
# 
# 15. 'is_repeated_guest': This column may contain a binary value indicating whether the guest is a repeated guest (e.g., 0 for not repeated and 1 for repeated).
# 
# 16. 'reserved_room_type': This column likely represents the room type initially reserved by the guest.
# 
# 17. 'assigned_room_type': This column may indicate the actual room type assigned to the guest.
# 
# 18. 'booking_changes': This column probably tracks the number of changes made to the booking.
# 
# 19. 'deposit_type': This column may specify the type of deposit made for the reservation, such as "No Deposit," "Non-Refundable," or "Refundable."
# 
# 20. 'days_in_waiting_list': This column may record the number of days the reservation was on the waiting list.
# 
# 21. 'customer_type': This column likely describes the customer type, such as "Transient" or "Group."
# 
# 22. 'adr': This column probably represents the average daily rate (price) of the reservation.
# 
# 23. 'required_car_parking_spaces': This column may indicate the number of car parking spaces requested by the guest.
# 
# 24. 'total_of_special_requests': This column likely records the total number of special requests made by the guest.
# 
# 25. 'reservation_status': This column probably indicates the status of the reservation, such as "Check-Out," "Canceled," or "No-Show."
# 

# In[2]:


dataframe = pd.read_csv('cleaned_dataframe.csv')


# In[3]:


dataframe.head()


# In[4]:


dataframe = dataframe.drop('Unnamed: 0', axis = 1)


# In[5]:


dataframe.head()


# In[6]:


dataframe.shape


# In[7]:


dataframe.isna().sum()


# In[8]:


dataframe.info()


# In[9]:


dataframe.columns 


# In[10]:


dataframe.hist(figsize=(15,15))
plt.show()


# In[11]:


plt.figure(figsize=(30,16))
sns.boxplot(data = dataframe[['lead_time','stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies', 'booking_changes', 'days_in_waiting_list', 'adr', 'required_car_parking_spaces', 'total_of_special_requests']])
plt.show()


# In[12]:


fig, axes=plt.subplots(1,3, figsize = (16,9))
sns.set_palette('husl')
ax = sns.boxplot(data = dataframe[['adr']], ax = axes[0])
ax = sns.boxplot(data = dataframe[['lead_time']], ax = axes[1])
ax = sns.boxplot(data = dataframe[['days_in_waiting_list']], ax = axes[2])
plt.show()


# In[ ]:





# In[ ]:




