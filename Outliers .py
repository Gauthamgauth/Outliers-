#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree  import DecisionTreeClassifier

from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer


# In[3]:


df = pd.read_csv("archive (3).zip")


# In[4]:


df.head()


# In[6]:


df.isnull()


# In[8]:


df.shape


# In[18]:


plt.figure(figsize=(9,5))
plt.subplot(1,2,1)
sns.histplot(df["cgpa"],kde=True)
plt.title("cgpa distrubutoion")

plt.subplot(1,2,2)
sns.histplot(df["package"],kde=True)
plt.title("package distribution")
plt.tight_layout()

plt.show()


# In[20]:


df["cgpa"].skew()


# In[21]:


df["package"].skew()


# In[22]:


print("cgpa mean",df["cgpa"].mean())
print("std cgpa",df["cgpa"].std())
print("min",df["cgpa"].min())
print("max",df["cgpa"].max())


# In[24]:


df["cgpa"].describe()


# In[25]:


sns.boxplot(df["cgpa"])


# In[26]:


sns.boxplot(df["package"])


# In[27]:


percentile25 = df["package"].quantile(0.25)
percentile75 = df["package"].quantile(0.75)


# In[28]:


percentile75


# In[29]:


percentile25


# In[30]:


iqr = percentile75 - percentile25


# In[31]:


iqr


# In[35]:


upper_limit = percentile75 + 1.5*iqr
lower_limit = percentile25 - 1.5*iqr


# In[38]:


print("upper limit",upper_limit)
print("lower limit",lower_limit)

