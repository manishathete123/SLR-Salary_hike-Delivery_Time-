# SLR-Salary_hike-Delivery_Time-
Simple Linear Regression
#!/usr/bin/env python
# coding: utf-8

# In[1]:


# impoort libraries
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf


# In[3]:


# import dataset
dataset=pd.read_csv('C:\\Users\\Dell\\Downloads\\Salary_Data.csv')
dataset


# # EDA and Data Visualization

# In[4]:


dataset.info()


# In[5]:


sns.distplot(dataset['YearsExperience'])


# In[6]:


sns.distplot(dataset['Salary'])


# # Correlation Analysis

# In[7]:


dataset.corr()


# In[8]:


sns.regplot(x=dataset['YearsExperience'],y=dataset['Salary'])


# # Model Building

# In[9]:


model=smf.ols("Salary~YearsExperience",data=dataset).fit()


# # Model Testing

# In[10]:


# Finding Cefficient Parameters
model.params


# In[11]:


# Finding Pvalues and tvalues
model.tvalues, model.pvalues


# In[12]:


# Finding Rsquared values
model.rsquared , model.rsquared_adj


# # Model Predictions

# In[13]:


# Manual prediction for say 3 Years Experience
Salary = (25792.200199) + (9449.962321)*(3)
Salary


# In[14]:


# Automatic Prediction for say 3 & 5 Years Experience 


# In[15]:


new_data=pd.Series([3,5])
new_data


# In[16]:


data_pred=pd.DataFrame(new_data,columns=['YearsExperience'])
data_pred


# In[17]:


model.predict(data_pred)

