#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


df = pd.read_excel(r'E:/ML Projects/BMI.xlsx')


# In[10]:


df.head()


# In[11]:


df.isnull().sum()


# In[12]:


df.dtypes


# In[13]:


df = df.drop(['Gender'],axis=1)


# In[14]:


df.head()


# In[17]:


x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.25,
                                                 random_state=0)


# In[20]:


from sklearn.linear_model import LinearRegression


# In[21]:


classfier = LinearRegression()


# In[22]:


classfier.fit(x_train,y_train)


# In[23]:


y_pred = classfier.predict(x_test)


# In[24]:


y_pred


# In[25]:


from sklearn.metrics import r2_score


# In[26]:


r2 = r2_score(y_test,y_pred)


# In[27]:


r2


# In[28]:


x.shape


# In[29]:


adj_r2 = (1-(1-r2)*(49/45))


# In[30]:


adj_r2


# In[ ]:




