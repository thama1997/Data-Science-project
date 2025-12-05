#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# In[2]:


df=pd.read_csv(r"F:\Internship Tasks\tested.csv")


# In[3]:


df


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# # drop cabin column

# In[6]:


df=df.drop('Cabin',axis=1)


# In[7]:


df.isnull().sum()


# # Fill Missing Values

# In[8]:


mean_age=df['Age'].mean()
df['Age']=df['Age'].fillna(mean_age)


# In[9]:


mean_fare=df['Fare'].mean()
df['Fare']=df['Fare'].fillna(mean_fare)


# In[10]:


df.isnull().sum()


# # EDA

# In[11]:


plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Pclass')
plt.title('Passenger Count by Pclass')
plt.xlabel('Pclass')
plt.ylabel('Count')
plt.show()


# In[12]:


plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Embarked')
plt.title('Passenger Count by Embarked')
plt.xlabel('Embarked')
plt.ylabel('Count')
plt.show()


# In[13]:


correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[14]:


plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='Fare', bins=30, kde=True)
plt.title('Fare Distribution')
plt.xlabel('Fare')
plt.ylabel('Count')
plt.show()


# # One Hot Encoding

# In[15]:


pclass_dummies = pd.get_dummies(df['Pclass'], prefix='Pclass')
df = pd.concat([df, pclass_dummies], axis=1)
df.drop('Pclass', axis=1, inplace=True)


# In[16]:


df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
df.drop('Name', axis=1, inplace=True)


# In[17]:


embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
df = pd.concat([df, embarked_dummies], axis=1)
df.drop('Embarked', axis=1, inplace=True)


# In[18]:


df


# In[19]:


sex_dummies = pd.get_dummies(df['Sex'], prefix='Sex')
df = pd.concat([df, sex_dummies], axis=1)
df.drop('Sex', axis=1, inplace=True)


# In[ ]:





# In[20]:


df


# In[ ]:





# In[21]:


X = df.drop('Survived', axis=1)
y = df['Survived']


# In[22]:


df


# In[23]:


X=df.drop(['Survived','Ticket','Title'],axis=1)


# In[24]:


y=df['Survived']


# In[25]:


X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2,random_state=42)


# # Logistic Regression

# In[26]:


logistic_regression = LogisticRegression(max_iter=1000)  # Increase max_iter if needed
logistic_regression.fit(X_train, y_train)
train_accuracy_lr = logistic_regression.score(X_train, y_train)
test_accuracy_lr = logistic_regression.score(X_test, y_test)
print(f'Logistic Regression - Training Accuracy: {train_accuracy_lr:.2f}, Testing Accuracy: {test_accuracy_lr:.2f}')


# #  Random Forest Classifier

# In[27]:


random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_train, y_train)

train_accuracy_rf = random_forest.score(X_train, y_train)
test_accuracy_rf = random_forest.score(X_test, y_test)


# In[28]:


print(f'Random Forest - Training Accuracy: {train_accuracy_rf:.2f}, Testing Accuracy: {test_accuracy_rf:.2f}')

