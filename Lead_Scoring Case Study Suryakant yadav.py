#!/usr/bin/env python
# coding: utf-8

# In[58]:


import warnings
warnings.filterwarnings('ignore')


# In[59]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[60]:


bike = pd.read_csv(r'C:\Users\suryakant.yadav\Desktop\New folder\day.csv')
bike.head()


# In[61]:


# Check the descriptive information
bike.info()


# In[62]:


bike.describe()


# In[63]:


# Check the shape of df

print(bike.shape)


# In[64]:


# percentage of missing values in each column
round(100*(bike.isnull().sum()/len(bike)), 2).sort_values(ascending=False)


# In[65]:


# row-wise null count percentage
round((bike.isnull().sum(axis=1)/len(bike))*100,2).sort_values(ascending=False)


# In[66]:


bike_dup = bike.copy()

# Checking for duplicates and dropping the entire duplicate row if any
bike_dup.drop_duplicates(subset=None, inplace=True)


# In[67]:


bike_dup.shape


# In[68]:


bike.shape


# In[69]:


#Create a copy of the  dataframe, without the 'instant' column, 

#as this will have unique values, and donot make sense to do a value count on it.

bike_dummy=bike.iloc[:,1:16]


# In[70]:


for col in bike_dummy:
    print(bike_dummy[col].value_counts(ascending=False), '\n\n\n')


# In[71]:


bike.columns


# In[72]:


bike_new=bike[['season', 'yr', 'mnth', 'holiday', 'weekday',
       'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed',
       'cnt']]


# In[73]:


bike_new.info()


# In[74]:


# Check the datatypes before convertion
bike_new.info()


# In[75]:


# Convert to 'category' data type

bike_new['season']=bike_new['season'].astype('category')
bike_new['weathersit']=bike_new['weathersit'].astype('category')
bike_new['mnth']=bike_new['mnth'].astype('category')
bike_new['weekday']=bike_new['weekday'].astype('category')


# In[76]:


bike_new.info()


# In[77]:


bike_new = pd.get_dummies(bike_new, drop_first=True)
bike_new.info()


# In[78]:


bike_new.shape


# In[79]:


# Check the shape before spliting

bike_new.shape


# In[80]:


# Check the info before spliting

bike_new.info()


# In[81]:


from sklearn.model_selection import train_test_split

# We should specify 'random_state' so that the train and test data set always have the same rows, respectively

np.random.seed(0)
df_train, df_test = train_test_split(bike_new, train_size = 0.70, test_size = 0.30, random_state = 333)


# In[82]:


df_train.info()


# In[83]:


df_train.shape


# In[84]:


df_test.info()


# In[85]:


df_test.shape


# In[86]:


df_train.info()


# In[87]:


df_train.columns


# In[88]:


# Create a new dataframe of only numeric variables:

bike_num=df_train[[ 'temp', 'atemp', 'hum', 'windspeed','cnt']]

sns.pairplot(bike_num, diag_kind='kde')
plt.show()


# In[89]:


df_train.info()


# In[90]:


# Build boxplot of all categorical variables (before creating dummies) againt the target variable 'cnt' 
# to see how each of the predictor variable stackup against the target variable.

plt.figure(figsize=(25, 10))
plt.subplot(2,3,1)
sns.boxplot(x = 'season', y = 'cnt', data = bike)
plt.subplot(2,3,2)
sns.boxplot(x = 'mnth', y = 'cnt', data = bike)
plt.subplot(2,3,3)
sns.boxplot(x = 'weathersit', y = 'cnt', data = bike)
plt.subplot(2,3,4)
sns.boxplot(x = 'holiday', y = 'cnt', data = bike)
plt.subplot(2,3,5)
sns.boxplot(x = 'weekday', y = 'cnt', data = bike)
plt.subplot(2,3,6)
sns.boxplot(x = 'workingday', y = 'cnt', data = bike)
plt.show()


# In[91]:


# Let's check the correlation coefficients to see which variables are highly correlated. Note:
# here we are considering only those variables (dataframe: bike_new) that were chosen for analysis

plt.figure(figsize = (25,20))
sns.heatmap(bike_new.corr(), annot = True, cmap="RdBu")
plt.show()


# In[92]:


from sklearn.preprocessing import MinMaxScaler


# In[93]:


scaler = MinMaxScaler()


# In[94]:


# Checking the values before scaling
df_train.head()


# In[95]:


df_train.columns


# In[96]:


# Apply scaler() to all the numeric variables

num_vars = ['temp', 'atemp', 'hum', 'windspeed','cnt']

df_train[num_vars] = scaler.fit_transform(df_train[num_vars])


# In[97]:


# Checking values after scaling
df_train.head()


# In[98]:


df_train.describe()


# In[99]:


y_train = df_train.pop('cnt')
X_train = df_train


# In[123]:


# Importing RFE and LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


# In[150]:


# Running RFE with the output number of the variable equal to 15
lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE (lm, n_features_to_select =15)           # running RFE
rfe = rfe.fit(X_train, y_train)


# In[151]:


list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[152]:


col = X_train.columns[rfe.support_]
col


# In[153]:


X_train.columns[~rfe.support_]


# In[154]:


# Creating X_test dataframe with RFE selected variables
X_train_rfe = X_train[col]


# In[156]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_rfe.columns
vif['VIF'] = [variance_inflation_factor(X_train_rfe.values, i) for i in range(X_train_rfe.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[157]:


import statsmodels.api as sm

# Add a constant
X_train_lm1 = sm.add_constant(X_train_rfe)

# Create a first fitted model
lr1 = sm.OLS(y_train, X_train_lm1).fit()


# In[158]:


# Check the parameters obtained

lr1.params


# In[159]:


# Print a summary of the linear regression model obtained
print(lr1.summary())


# In[161]:


X_train_new = X_train_rfe.drop(["atemp"], axis = 1)


# In[162]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_new.columns
vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[163]:


# Add a constant
X_train_lm2 = sm.add_constant(X_train_new)

# Create a first fitted model
lr2 = sm.OLS(y_train, X_train_lm2).fit()


# In[164]:


# Check the parameters obtained

lr2.params


# In[165]:


# Print a summary of the linear regression model obtained
print(lr2.summary())


# In[166]:


X_train_new = X_train_new.drop(["hum"], axis = 1)


# In[167]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_new.columns
vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[168]:


# Add a constant
X_train_lm3 = sm.add_constant(X_train_new)

# Create a first fitted model
lr3 = sm.OLS(y_train, X_train_lm3).fit()


# In[169]:


lr3.params


# In[170]:


# Print a summary of the linear regression model obtained
print(lr3.summary())


# In[171]:


X_train_new = X_train_new.drop(["season_3"], axis = 1)


# In[172]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_new.columns
vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[173]:


# Add a constant
X_train_lm4 = sm.add_constant(X_train_new)

# Create a first fitted model
lr4 = sm.OLS(y_train, X_train_lm4).fit()


# In[174]:


# Check the parameters obtained

lr4.params


# In[175]:


# Print a summary of the linear regression model obtained
print(lr4.summary())


# In[176]:


X_train_new = X_train_new.drop(["mnth_10"], axis = 1)


# In[177]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_new.columns
vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[178]:


# Add a constant
X_train_lm5 = sm.add_constant(X_train_new)

# Create a first fitted model
lr5 = sm.OLS(y_train, X_train_lm5).fit()


# In[179]:


# Check the parameters obtained

lr5.params


# In[180]:


# Print a summary of the linear regression model obtained
print(lr5.summary())


# In[181]:


X_train_new = X_train_new.drop(["mnth_3"], axis = 1)


# In[182]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_new.columns
vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[183]:


# Add a constant
X_train_lm6 = sm.add_constant(X_train_new)

# Create a first fitted model
lr6 = sm.OLS(y_train, X_train_lm6).fit()


# In[184]:


# Check the parameters obtained

lr6.params


# In[185]:


# Print a summary of the linear regression model obtained
print(lr6.summary())


# In[186]:


y_train_pred = lr6.predict(X_train_lm6)


# In[187]:


res = y_train-y_train_pred
# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((res), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)                         # X-label


# In[188]:


bike_new=bike_new[[ 'temp', 'atemp', 'hum', 'windspeed','cnt']]

sns.pairplot(bike_num, diag_kind='kde')
plt.show()


# In[189]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_new.columns
vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[190]:


# Apply scaler() to all numeric variables in test dataset. Note: we will only use scaler.transform, 
# as we want to use the metrics that the model learned from the training data to be applied on the test data. 
# In other words, we want to prevent the information leak from train to test dataset.

num_vars = ['temp', 'atemp', 'hum', 'windspeed','cnt']

df_test[num_vars] = scaler.transform(df_test[num_vars])


# In[191]:


df_test.head()


# In[192]:


df_test.describe()


# In[193]:


y_test = df_test.pop('cnt')
X_test = df_test
X_test.info()


# In[194]:


#Selecting the variables that were part of final model.
col1=X_train_new.columns
X_test=X_test[col1]
# Adding constant variable to test dataframe
X_test_lm6 = sm.add_constant(X_test)
X_test_lm6.info()


# In[195]:


# Making predictions using the final model (lr6)

y_pred = lr6.predict(X_test_lm6)


# In[196]:


# Plotting y_test and y_pred to understand the spread

fig = plt.figure()
plt.scatter(y_test, y_pred, alpha=.5)
fig.suptitle('y_test vs y_pred', fontsize = 20)              # Plot heading 
plt.xlabel('y_test', fontsize = 18)                          # X-label
plt.ylabel('y_pred', fontsize = 16) 
plt.show()


# In[197]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[198]:


# We already have the value of R^2 (calculated in above step)

r2=0.8203092200749708


# In[199]:


# Get the shape of X_test
X_test.shape


# In[200]:


# n is number of rows in X

n = X_test.shape[0]


# Number of features (predictors, p) is the shape along axis 1
p = X_test.shape[1]

# We find the Adjusted R-squared using the formula

adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
adjusted_r2


# In[ ]:




