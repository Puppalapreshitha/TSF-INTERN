#!/usr/bin/env python
# coding: utf-8

# GRIP - The Sparks Foundation - #GRIPOCT21
# Data Science and Business Analytics
# TASK-1:Prediction using Supervised ML
# Author: Jampala Sri Naga Sai
# Data : http://bit.ly/w-data¶

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


url="http://bit.ly/w-data"
data=pd.read_csv(url)
data


# In[3]:


data.shape


# In[4]:


data.info()


# In[5]:


data.describe()


# In[6]:


sns.scatterplot(x=data['Hours'],y=data['Scores'])


# In[7]:


sns.regplot(x=data['Hours'],y=data['Scores'])


# In[8]:


X=data[['Hours']]
Y=data['Scores']


# In[9]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)


# In[10]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)


# In[11]:


Y_pred=regressor.predict(X_test)


# In[12]:


f=pd.DataFrame({'Actual':Y_test,'Predicted':Y_pred})
f


# ## What will be the predicted score if a student studies for 9.25 hrs/day?

# In[11]:


hours=9.25
ans=regressor.predict([[hours]])
print("No. of Hours={}".format(hours))
print("Predicted Score={}".format(ans[0]))

