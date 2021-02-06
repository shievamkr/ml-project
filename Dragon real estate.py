#!/usr/bin/env python
# coding: utf-8

# ## dragon real estate-price predictor

# In[1]:


import pandas as pd


# In[2]:


housing=pd.read_csv("data.csv")


# In[3]:


housing.head()


# In[4]:


housing.info


# In[5]:


housing.info()


# In[6]:


housing['CHAS'].value_counts()


# In[7]:


housing.describe()


# In[8]:


housing.describe()


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


import matplotlib.pyplot as plt


# In[11]:


#housing.hist(bins=50,figsize=(20,15))


# # #train-test splitting

# In[12]:


import numpy as np
#for learning purpose
def split_train_test(data,test_ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices,],data.iloc[test_indices]


# In[13]:


#train_set,test_set = split_train_test(housing,0.2)


# In[14]:


print(f"rows in train set:-{len(train_set)}\n rows in test set:-{len(test_set)}\n")


# In[15]:


from sklearn.model_selection import train_test_split
train_set,test_set = train_test_split(housing,test_size=0.2,random_state=42)
print(f"rows in train set:-{len(train_set)}\n rows in test set:-{len(test_set)}\n")


# In[16]:


#stratifiedshufflesplit is so important for sensetive data like CHAS
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing['CHAS']):
    strat_train_set=housing.loc[train_index]
    strat_test_set=housing.loc[test_index]


# In[17]:


strat_test_set['CHAS'].value_counts()


# In[18]:


strat_train_set['CHAS'].value_counts()


# In[19]:


housing = strat_train_set.copy()


# ## looking for correlations
# 

# In[20]:


from  pandas.plotting import scatter_matrix
attributes = ['MEDV','RM','ZN','LSTAT']
scatter_matrix(housing[attributes],figsize=(12,8))


# In[21]:


housing.plot(kind='scatter',x='RM',y='MEDV')


# ## Trying out attribute combinations

# In[22]:


#housing['TAXRM']=housing['TAX']/housing['RM']


# In[23]:


corr_matrix= housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# ## missing attributes

# In[24]:


#to take care of missing attributes you have three options:-
#  1.get rid of the missing data points
#  2.get rid of the whole attribute
#  3.set the value to some value(0,mean or median)


# In[25]:


a= housing.dropna(subset=['MEDV'])  #option 1
a.shape


# In[26]:


housing.drop("MEDV",axis=1).shape  #option2
#note that there is no change in original dataset.


# In[27]:


mn=housing['MEDV'].median()


# In[28]:


mn


# In[29]:


housing['MEDV'].fillna(mn) #option 3
#note that there is no change in original dataset.


# In[30]:


housing.shape


# In[41]:


from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy=float('median'))
imputer.fit(housing)


# In[37]:


x=imputer.transform(housing)
housing_tr = pd.DataFrame(x,columns=housing.coloumns)
housing_tr.describe()


# ## scikit-learn design

# primarily three types of objects 
# 1. Estimators - it estimates some parameter based on a dataset eg. imputer
# it has a fit method and transform methods.
# fit method- fits the dataseet and calculates internal parameters
# 
# 2. transformers - transforms methid takes input and returns output based on 
# the learning from fit(). it also has a convinance function called fit_transform(_
# 
# 3. predictors - linearregrassion model is an example of predictor .fit() and 
# predict()  are two common functions . it also gives score() function which will 
# evaluate the prediction.

# ## feature scaling

# primarily two types of feature scaling methods:
#     1.min-max scaling (normalization)
#     (value-min)/(max-min)
#     sklearn provides class called MinMaxScaler for this.
#     
#     2. standradization:-
#     (value-min)/std
#     sklearn provides a class called standard standardScaler for this

# #  creating a pipeline

# In[35]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([('imputer',SimpleImputer(strategy="median")),
                       ('std_scaler',StandardScaler())])


# In[38]:


housing_num_tr=my_pipeline.fit_transform(housing_tr)


# In[ ]:




