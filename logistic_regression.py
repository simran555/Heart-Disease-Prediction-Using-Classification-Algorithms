#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


df=pd.read_csv('D:\datasets\health\heart.csv')


# In[4]:


df.head()


# In[5]:


import numpy as nm
import matplotlib.pyplot as mtp  


# In[6]:


#extracting dependent and independent variable 
x=df.iloc[:,0:13]


# In[7]:


print(x.head())


# In[8]:


y=df.iloc[:,13]


# In[9]:


print(y.head())


# In[10]:


#split the dataset into training and test dataset
from sklearn.model_selection import train_test_split 


# In[11]:


x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0) 
print(x_train.head(),y_train.head())
print(x_test.head(),y_test.head())


# In[12]:


#we need to scaling for accurate predictions ,only for indepedent variables because target is 0 or 1
from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test) 


# In[13]:


print(x_train)
print(x_test)


# In[14]:


#fitting to training set
from sklearn.linear_model import LogisticRegression  
classifier= LogisticRegression(random_state=0)  
classifier.fit(x_train, y_train) 


# In[15]:


#Predicting the test set result  
y_pred= classifier.predict(x_test)
print(y_pred)


# In[16]:


#making confusion matrix to test the accuracy
from sklearn.metrics import confusion_matrix


# In[17]:


conf_mat=confusion_matrix(y_test,y_pred)


# In[18]:


print(conf_mat)


# In[ ]:


#222 correct predictions
#35 incorrect predictions
#precision=79%
#recall=90%
#accuracy=

