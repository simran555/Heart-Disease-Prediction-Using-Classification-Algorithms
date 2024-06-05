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


from sklearn.svm import SVC # "Support vector classifier"  
classifier = SVC(kernel='linear', random_state=0)  
classifier.fit(x_train, y_train) 


# In[7]:


#Predicting the test set result  
y_pred= classifier.predict(x_test) 
y_pred


# In[8]:


#Creating the Confusion matrix  
from sklearn.metrics import confusion_matrix  
cm= confusion_matrix(y_test, y_pred)
cm


# In[ ]:


#99+122=221 correct predictions
#24+12=36 incorrect predictions
#precision=80%
#recall=

