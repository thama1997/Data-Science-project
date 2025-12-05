#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv(r"F:\Internship Tasks\IRIS.csv")


# In[3]:


df


# In[4]:


df.isnull().sum()


# # EDA

# In[5]:


sns.pairplot(df, hue='species', palette='Set1')
plt.title('Pairplot of Iris Dataset')
plt.show()



# In[6]:


plt.figure(figsize=(12, 6))
for i, feature in enumerate(df.columns[:-1]):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(x='species', y=feature, data=df, palette='Set1')
    plt.title(f'Boxplot of {feature} by Species')
plt.tight_layout()
plt.show()



# In[7]:


plt.figure(figsize=(12, 6))
for i, feature in enumerate(df.columns[:-1]):
    plt.subplot(2, 2, i + 1)
    sns.violinplot(x='species', y=feature, data=df, palette='Set1')
    plt.title(f'Violinplot of {feature} by Species')
plt.tight_layout()
plt.show()



# In[8]:


correlation_matrix = df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[9]:


df['species'].unique()


# In[10]:


X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # SVM Model

# In[11]:


svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
print("SVM Model Accuracy:", svm_accuracy)


# # RandomForestClassifier Model 

# In[12]:


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("Random Forest Model Accuracy:", rf_accuracy)


# In[ ]:




