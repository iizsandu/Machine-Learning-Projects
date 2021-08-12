#!/usr/bin/env python
# coding: utf-8

# # Sandip Shaw

# ## Machine Learning Short Assignment 

# ### Goal : To create a python notebook (jupyter) and predict value of variable (target) for next 6 quarters using Pycaret package 

#    

# 

# In[397]:


import pandas as pd


# In[398]:


# To view every Rows

pd.set_option('display.max_rows',None)


# In[399]:


# Read CSV file

df = pd.read_csv('fred_quarterly.csv')
df.head()


# In[400]:


df.tail()


# In[401]:


df.shape


# In[402]:


# Checking for Null Values

df.isnull().sum()


# In[403]:


# Removing the only record with null value

new_df = df.dropna()
new_df.tail()


# In[404]:


# Information about the dataframe

new_df.info()


# In[405]:


# Descriptive Statistics

new_df.describe()


# In[406]:


new_df['quarter'].value_counts().sum()


# In[407]:


cols = new_df.columns
cols


# In[408]:


# Checking for Unique Values

for i in range(len(new_df)):
    print('Unique Values in ',cols[i])
    print(new_df[cols[i]].value_counts().sum())


# In[409]:


# Removing the Year Value from the quarters

new_df['quarter'] = new_df['quarter'].str[4:]
new_df.head()


# In[448]:


# Removing the date and month variable as they have no correlation with the target

new_df1 = new_df.drop(['date','month'],axis=1)
new_df1.head()


#   

# ### According to the question, we need to predict values of 6 records so we split the given dataframe into 79 and 6 records into train and test dataframe

# In[411]:


train_df = new_df1.iloc[:-6,:]
train_df.head()


# In[412]:


train_df.tail()


# In[415]:


test_df = new_df1.iloc[79:85,:]
test_df = test_df.drop(['Target'],axis=1)
test_df


#   

# In[416]:


# DataFrame for Visualization

df_vis = train_df.drop(['quarter'],axis=1)
df_vis.head()


# ## DATA VISUALIZATION 

# In[417]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[418]:


# heatmap using correkation matrix

plt.figure(figsize=(10,10))
sns.heatmap(df_vis.corr(),cmap='Oranges')


# ### There is no correlation between variables
# 

# In[419]:


# PairPlot

sns.pairplot(df_vis)


# In[420]:


# Pairplot using Target as Hue

sns.pairplot(df_vis,hue='Target')


# In[421]:


# histogram

sns.histplot(df_vis['Target'],kde=True)


#    

#   

#   

# In[422]:


train_df['quarter'].value_counts()


# In[423]:


Q1total = train_df.loc[new_df['quarter']=='Q1','Target'].sum()
Q1total


# In[424]:


Q2total = train_df.loc[new_df['quarter']=='Q2','Target'].sum()
Q2total


# In[425]:


Q3total = train_df.loc[new_df['quarter']=='Q3','Target'].sum()
Q3total


# In[426]:


Q4total = train_df.loc[new_df['quarter']=='Q4','Target'].sum()
Q4total


# We can clearly see Quarter 1 has slightly more overall target than Q1,Q2,Q3 which has negligible difference among each other

# In[427]:


train_df.info()


# ## One Hot Encoding (for object data type variable Quarter)

# In[428]:


from sklearn.preprocessing import OneHotEncoder


# In[429]:


oh_enc = OneHotEncoder(sparse=False,drop='first')


# In[430]:


oh_enc_arr = oh_enc.fit_transform(train_df[['quarter']])
oh_enc_arr


# In[431]:


oh_enc_arr2 = oh_enc.fit_transform(test_df[['quarter']])
oh_enc_arr2


# In[432]:


dummy_df = pd.get_dummies(train_df)
dummy_df.head()


# In[433]:


dummy_df2 = pd.get_dummies(test_df)
dummy_df2.head()


# In[ ]:





# In[434]:


oh_enc_df = pd.DataFrame(oh_enc_arr, columns=['quarter_Q2','quarter_Q3','quarter_Q4'])
oh_enc_df.head()


# In[435]:


oh_enc_df2 = pd.DataFrame(oh_enc_arr2, columns=['quarter_Q2','quarter_Q3','quarter_Q4'])
oh_enc_df2 = oh_enc_df2.set_index([pd.Index([79,80,81,82,83,84])])
oh_enc_df2


# In[436]:


final_df = pd.DataFrame.join(train_df,oh_enc_df)
final_df.head()


# In[437]:


final_test_df = pd.DataFrame.join(test_df,oh_enc_df2)
final_test_df


# In[438]:


final_df2 = final_df.drop(['quarter'],axis=1)


# In[439]:


final_df2.head()


# In[440]:


final_df2.tail()


# ## Using the Pycaret module

# In[441]:


from pycaret.regression import *


# In[442]:


clf1 = setup(data = final_df2,target = 'Target')


# In[443]:


compare_models()


# In[444]:


reg_model = create_model("et")


# In[445]:


tuned_model = tune_model(reg_model,n_iter=50,optimize='MAE')


# In[446]:


et_final = finalize_model(tuned_model)


# ### Predicting on the test dataframe we created earlier

# In[447]:


predictions = predict_model(et_final, data = final_test_df)
predictions


# In[ ]:




