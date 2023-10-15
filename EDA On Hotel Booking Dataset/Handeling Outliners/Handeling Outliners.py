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


# ## Importing Dataset :

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


dataframe = pd.read_csv('cleaned_dataframe')


# In[3]:


dataframe.head()


# In[4]:


dataframe = dataframe.drop('Unnamed: 0', axis = 1)


# In[5]:


dataframe.head()


# In[6]:


dataframe.columns


# In[7]:


new_dataframe = dataframe.copy()


# In[8]:


new_dataframe.loc[new_dataframe.stays_in_weekend_nights > 5, 'stays_in_weekend_nights'] = 5
new_dataframe.loc[new_dataframe.stays_in_week_nights > 10,'stays_in_week_nights'] = 10


# In[9]:


new_dataframe.loc[new_dataframe.adults > 4,'adults'] = 4
new_dataframe.loc[new_dataframe.children > 3,'children'] = 3
new_dataframe.loc[new_dataframe.babies > 3,'babies'] = 3


# In[10]:


new_dataframe.loc[new_dataframe.required_car_parking_spaces > 3,'required_car_parking_spaces'] = 3


# In[11]:


new_dataframe.loc[new_dataframe.booking_changes > 5,'booking_changes'] = 5


# In[12]:


new_dataframe.head().T


# ### Applying IQR Methode :

# The IQR (Interquartile Range) method is a statistical technique used to measure the spread or dispersion of data in a dataset. It is often used in data analysis and is especially useful for identifying and handling outliers. The IQR method is based on quartiles, which divide a dataset into four equal parts, with three quartiles in between.
# 
# Here's how the IQR method works:
# 
# 1. Arrange the data in ascending order.
# 
# 2. Find the first quartile (Q1), which is the median of the lower half of the data (the 25th percentile).
# 
# 3. Find the third quartile (Q3), which is the median of the upper half of the data (the 75th percentile).
# 
# 4. Calculate the IQR by subtracting Q1 from Q3:
#    IQR = Q3 - Q1
# 
# 5. Identify potential outliers by defining a lower bound and an upper bound. Values below the lower bound or above the upper bound are considered outliers.
#    - Lower Bound = Q1 - (1.5 * IQR)
#    - Upper Bound = Q3 + (1.5 * IQR)
# 
# 6. Any data point below the lower bound or above the upper bound is considered an outlier and may require further investigation or treatment, depending on the context of the analysis.
# 

# In[15]:


for col in ['lead_time','days_in_waiting_list']:
  lower_cap, q1, q3, upper_cap, median = dataframe[col].quantile([0.01,0.25,0.75,0.99,0.5])
  lower_limit = q1 - 1.5*(q3-q1)
  upper_limit = q3 + 1.5*(q3-q1)

  new_dataframe[col] = np.where(new_dataframe[col] > upper_limit, median,np.where(
                         new_dataframe[col] < lower_limit,median,np.where(
                         new_dataframe[col] < lower_cap,lower_cap,np.where(
                         new_dataframe[col] > upper_cap,upper_cap,new_dataframe[col]))))


# In[16]:


new_dataframe.head().T


# Removinh the outliners from 'adr' column.

# In[17]:


new_dataframe.drop(new_dataframe[new_dataframe['adr'] > 5000].index, inplace = True)
new_dataframe.drop(new_dataframe[new_dataframe['adr'] <= 0].index, inplace = True)


# Categorical features with numerical values.

# In[18]:


categorical_features=["hotel","arrival_date_month","meal","country","market_segment","is_canceled",
                      "distribution_channel","reserved_room_type","assigned_room_type","deposit_type",
                      "customer_type","reservation_status","is_repeated_guest",'same_room',
                      'arrival_date_year']

numeric_features = [i for i in dataframe.columns if i not in categorical_features]
print(numeric_features)


# In[24]:


for col in numeric_features:
  fig, ax =plt.subplots(1,2, constrained_layout=True)
  fig.set_size_inches(15, 3)
  sns.histplot(dataframe[col], ax=ax[0]).set(title="Before")
  sns.boxplot(dataframe[col], ax=ax[1]).set(title="Before")

