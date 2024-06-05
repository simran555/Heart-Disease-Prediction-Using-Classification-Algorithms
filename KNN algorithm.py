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


#extracting dependent and independent variable 
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


#we need to scaling for accurate predictions ,only for indepedent variables because target is 0 or 1
from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test) 
print(x_train)
print(x_test)


# In[6]:


#fitting KNN classifier to our training data
from sklearn.neighbors import KNeighborsClassifier  
classifier= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  #we are taking the k-value as 5 ,p=2 means euclidean distance
classifier.fit(x_train, y_train)


# In[29]:


#now lets predict the output 
y_pred= classifier.predict(x_test)


# In[30]:


print(y_pred)


# In[10]:


print(x_test)


# In[22]:


array=nm.array([x_test[0]])
array


# In[25]:


#y_pred= classifier.predict(array)


# In[27]:


#y_pred[0] #output of the first row in the x_test 2darray says the person has heart attack


# In[31]:


#lets find the confusion matrix
from sklearn.metrics import confusion_matrix  
cm= confusion_matrix(y_test, y_pred)
cm


# In[ ]:


#106+17=123 correct predictions
#17+14=31 incorrect predictions
#precision=86%
#recall=

