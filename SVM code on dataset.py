#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


mushroom = pd.read_csv('data/mushrooms.csv')
mushroom.head()


# In[5]:


X = mushroom.drop('class', axis=1) # features

y = mushroom['class'] # labels


X.shape, y.shape


# In[6]:


y[:5]


# In[7]:


y.value_counts()


# In[10]:


from sklearn.preprocessing import LabelEncoder

X_enc = X.copy()
for col in X.columns:
    lb = LabelEncoder()  # shift + Tab
    X_enc[col] = lb.fit_transform(X[col].values)
    
X_enc.head()    


# In[11]:


X.shape, X_enc.shape


# In[12]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

#dividing our data
X_train, X_test, y_train, y_test = train_test_split(X_enc, y)

# The sequence of X_train, X_test, y_train, y_test is important becaucse it stores the correponding values


svc = SVC() # NO hyperparameter given means it will use default values(i.e - c = 1.0 and kernel = 'rbf')

svc.fit(X_train, y_train)


# In[13]:


svc.score(X_train, y_train) # train <a class="autolink" title="Dataset" href="https://course.aiadventures.in/mod/resource/view.php?id=2084">dataset</a>


# In[14]:


svc.score(X_test, y_test) # test <a class="autolink" title="Dataset" href="https://course.aiadventures.in/mod/resource/view.php?id=2084">dataset</a>


# In[ ]:




