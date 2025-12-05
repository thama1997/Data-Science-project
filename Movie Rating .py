#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


# # Load Data 

# In[2]:


df = pd.read_csv(r"F:\Internship Tasks\IMDb Movies India.csv", encoding='iso-8859-1')


# In[3]:


df


# In[4]:


df.info()


# # Checking missing values

# In[5]:


df.isnull().sum()


# # Fill missing values
# 

# In[6]:


def extract_year(year_str):
    if isinstance(year_str, str):
        try:
            return int(year_str.strip('()'))
        except ValueError:
            pass
    return year_str

df['Year'] = df['Year'].apply(extract_year)

df['Year'].fillna(df['Year'].median(), inplace=True)

print(df.isnull().sum())


# In[7]:


def extract_numeric(duration_str):
   
    matches = re.findall(r'\d+', str(duration_str))
    if matches:
        return int(matches[0])  
    else:
        return np.nan  

df['Duration'] = df['Duration'].apply(extract_numeric)


mean_duration = df['Duration'].mean()
df['Duration'].fillna(mean_duration, inplace=True)


# In[8]:


df['Genre'].fillna(df['Genre'].mode()[0], inplace=True)
df['Rating'].fillna(df['Rating'].mean(), inplace=True)


# In[9]:


def convert_to_numeric(votes_str):
    if isinstance(votes_str, str):
        try:
           
            return int(votes_str.replace(',', ''))
        except ValueError:
            pass
    return np.nan

df['Votes'] = df['Votes'].apply(convert_to_numeric)

median_votes = df['Votes'].median()
df['Votes'].fillna(median_votes, inplace=True)


# In[10]:


df['Director'].fillna(df['Director'].mode()[0], inplace=True)
df['Actor 1'].fillna('Unknown', inplace=True)
df['Actor 2'].fillna('Unknown', inplace=True)
df['Actor 3'].fillna('Unknown', inplace=True)
print(df.isnull().sum())


# # EDA

# In[11]:


plt.figure(figsize=(10, 6))
sns.histplot(df['Year'].dropna(), bins=20, kde=True)
plt.title('Distribution of Movies by Year')
plt.xlabel('Year')
plt.ylabel('Count')
plt.show()


# In[12]:


plt.figure(figsize=(12, 6))
top_genres = df['Genre'].value_counts().head(10)
sns.barplot(x=top_genres.values, y=top_genres.index, orient='h')
plt.title('Top 10 Most Common Movie Genres')
plt.xlabel('Count')
plt.ylabel('Genre')
plt.show()


# In[13]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x='Rating', y='Votes', data=df, alpha=0.5)
plt.title('Scatter Plot of Rating vs. Votes')
plt.xlabel('Rating')
plt.ylabel('Votes')
plt.show()


# # Apply Label Encoder

# In[14]:


label_encoder = LabelEncoder()
df['Genre'] = label_encoder.fit_transform(df['Genre'])
df['Director'] = label_encoder.fit_transform(df['Director'])
df['Actor 1'] = label_encoder.fit_transform(df['Actor 1'])
df['Actor 2'] = label_encoder.fit_transform(df['Actor 2'])
df['Actor 3'] = label_encoder.fit_transform(df['Actor 3'])


# In[15]:


df


# # Features and target variable and split data into 80% train, 20% test

# In[16]:


X = df.drop(columns=['Rating','Name']) 
y = df['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# #  Linear Regression

# In[17]:


linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
linear_reg_predictions = linear_reg.predict(X_test)
linear_reg_rmse = mean_squared_error(y_test, linear_reg_predictions, squared=False)
linear_reg_r2 = r2_score(y_test, linear_reg_predictions)


# #  Random Forest Regressor

# In[18]:


rf_reg = RandomForestRegressor(random_state=42)
rf_reg.fit(X_train, y_train)
rf_reg_predictions = rf_reg.predict(X_test)
rf_reg_rmse = mean_squared_error(y_test, rf_reg_predictions, squared=False)
rf_reg_r2 = r2_score(y_test, rf_reg_predictions)


# #  Gradient Boosting Regressor

# In[19]:


gb_reg = GradientBoostingRegressor(random_state=42)
gb_reg.fit(X_train, y_train)
gb_reg_predictions = gb_reg.predict(X_test)
gb_reg_rmse = mean_squared_error(y_test, gb_reg_predictions, squared=False)
gb_reg_r2 = r2_score(y_test, gb_reg_predictions)


# #  evaluation metrics for each model

# In[20]:


print(f'Linear Regression RMSE: {linear_reg_rmse:.2f} {linear_reg_r2:.2f}')
print(f'Random Forest RMSE: {rf_reg_rmse:.2f} {rf_reg_r2:.2f}')
print(f'Gradient Boosting RMSE: {gb_reg_rmse:.2f} {gb_reg_r2:.2f}')


# In[ ]:




