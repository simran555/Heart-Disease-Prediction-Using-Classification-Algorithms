#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as nm
import matplotlib.pyplot as mtp


# In[2]:


df=pd.read_csv('D:\datasets\health\heart.csv')
df.head()


# In[3]:


x=df.iloc[:,0:13]
print(x.head())
y=df.iloc[:,13]
print(y.head())


# In[4]:


from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0) 
print(x_train.head(),y_train.head())
print(x_test.head(),y_test.head())


# In[5]:


from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test) 
print(x_train)
print(x_test)


# In[6]:



from sklearn.naive_bayes import BernoulliNB  
classifier = BernoulliNB()  
classifier.fit(x_train, y_train)  


# In[7]:


y_pred = classifier.predict(x_test) 


# In[8]:


print(y_pred)


# In[9]:


# Making the Confusion Matrix  
from sklearn.metrics import confusion_matrix  
cm = confusion_matrix(y_test, y_pred)
cm


# In[10]:


from sklearn.naive_bayes import GaussianNB  
classifier = GaussianNB()  
classifier.fit(x_train, y_train)  


# In[11]:


y_pred = classifier.predict(x_test) 


# In[12]:


print(y_pred)


# In[13]:


# Making the Confusion Matrix  
from sklearn.metrics import confusion_matrix  
cm = confusion_matrix(y_test, y_pred)
cm


# In[ ]:


#219 correct predictions
#38 incorrect predictions
#precision=79%
#recall=88%
#accuracy=85%

