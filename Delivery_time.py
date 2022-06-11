#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf


# In[6]:


# import dataset
dataset=pd.read_csv('C:\\Users\\Dell\\Downloads\\Delivery_time.csv')
dataset


# # EDA and Data Visualization
# 

# In[7]:


dataset.info()


# In[8]:


sns.distplot(dataset['Delivery Time'])


# In[9]:


sns.distplot(dataset['Sorting Time'])


# # Feature Engineering

# In[10]:


# Renaming Columns
dataset=dataset.rename({'Delivery Time':'delivery_time', 'Sorting Time':'sorting_time'},axis=1)
dataset


# # Correlation Analysis

# In[11]:


dataset.corr()


# In[12]:


sns.regplot(x=dataset['sorting_time'],y=dataset['delivery_time'])


# # Model Building

# In[13]:


model=smf.ols("delivery_time~sorting_time",data=dataset).fit()


# # Model Testing

# In[14]:


# Finding Coefficient parameters
model.params


# In[15]:


# Finding tvalues and pvalues
model.tvalues , model.pvalues


# In[16]:


# Finding Rsquared Values
model.rsquared , model.rsquared_adj


# # Model Predictions

# In[17]:


# Manual prediction for say sorting time 5
delivery_time = (6.582734) + (1.649020)*(5)
delivery_time


# In[18]:


# Automatic Prediction for say sorting time 5, 8
new_data=pd.Series([5,8])
new_data


# In[19]:


data_pred=pd.DataFrame(new_data,columns=['sorting_time'])
data_pred


# In[20]:


model.predict(data_pred)


# In[ ]:




