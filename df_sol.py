# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 19:44:00 2018

@author: PAVEETHRAN
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


train=pd.read_csv('train.csv')
full_c=pd.read_csv('fulfilment_center_info.csv')
meal_in=pd.read_csv('meal_info.csv')
test=pd.read_csv('test.csv')
samp=pd.read_csv('sample_submission_hSLSoT6.csv')

train.describe()

test.loc[(test['checkout_price'] > 300) & (test['checkout_price'] < 500), 'Cat'] = 4
#38 exc in dictionaries
#Train
res=pd.merge(train,full_c,on='center_id')
train=pd.merge(res,meal_in,on='meal_id')

#Test
res1=pd.merge(test,full_c,on='center_id')
test=pd.merge(res1,meal_in,on='meal_id')

c=0
for i in range(len(test)):
    if(train['id'][i]==test['id'][i]):
        c=c+1
print(c)    
#NO COMMON VALUES
y=train['num_orders']
train.drop(['num_orders','id'],1,inplace=True)
y.dtypes
id_ind=test['id']
test=test.drop(['id'],1)

train.dtypes
test.dtypes
cat_fea=train.dtypes[train.dtypes=='object'].index
int_fea=train.dtypes[train.dtypes=='int'].index
float_fea=train.dtypes[train.dtypes=='float'].index


train.isnull().sum()
test.isnull().sum()

#total=pd.concat([train,test])
#
#total.isnull().sum()
#total.dtypes

#t=total
#cat_enc=pd.get_dummies(t[cat_fea],prefix=cat_fea).astype(int)
#
#t=pd.concat([t,cat_enc],1)
#t.drop(cat_fea,axis=1,inplace=True)
#total=t

#total.dtypes
#int_fea=total.dtypes[total.dtypes=='int'].index
#float_fea=total.dtypes[total.dtypes=='float'].index

#not_needed=['id']
#total=total.drop(not_needed,axis=1)


#from sklearn.preprocessing import MinMaxScaler
#mms=MinMaxScaler(feature_range=(0,1))
#total_sc=mms.fit_transform(total)
#y=np.array(y).reshape(-1,1)
#y_sc=mms.fit_transform(y)
#
#xtrain=total_sc[:len(train)]
#xtest=total_sc[len(train):]

#FEATURE ENGINEERING
#xtrain=total[:len(train)]
#xtest=total[len(train):]


#
#from sklearn.decomposition import PCA
#pca=PCA()
#pca.fit(train)
#exp_var=pca.explained_variance_ratio_

corrmat=train.corr().abs()
import seaborn as sns
sns.heatmap(corrmat,square=True,robust=True)

#DROPPING THE HIGLY CORRELATED DATA
upper=corrmat.where(np.triu(np.ones(corrmat.shape),k=1)==1)
corr_col=[column for column in upper.columns if any(upper[column]>0.6)]
train=train.drop(corr_col,1)
test=test.drop(corr_col,1)

train.dtypes

#AGGREGATE AND GROUPBY FEATURES


#
##VISUALIZE OUTPUT VARIABLE
#plt.plot(y)
#y=np.array(y)
#y.shape
#y.max()
#y.min()
#
#y_sc.max()
#y_sc.min()


sns.distplot(y,fit='norm')#....here norm gives the normalized data grapgh
plt.ylabel('frequency in float')

#AFTER SCALEING ...(ALREADY DID..USE Y_SC)
sns.distplot(y_sc,fit='norm')#....here norm gives the normalized data grapgh
plt.ylabel('frequency in float')

train_no_cat=train.drop(cat_fea,1)

from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(verbose=25,n_estimators=50)
rf.fit(train_no_cat,y)


#y_pred=rf.predict(xtest)
#y_pred=y_pred.reshape(-1,1)
##y_pred.shape
#
#y_pred=mms.inverse_transform(y_pred)
#
#id=test['id']
#id=id.reshape()

#
rf_fea_importances=pd.DataFrame(rf.feature_importances_
,index=train_no_cat.columns,columns=['Importances']).sort_values('Importances',ascending=False)


rank=np.argsort(-rf.feature_importances_)
rf_fea_importances.reset_index(inplace=True)
rf_fea_importances.iloc[1][0]

train_new=pd.DataFrame()
for i in range(0,7):
    indx=rf_fea_importances.iloc[i][0]
    print(indx)
    train_new[indx]=train[indx]
 
test_new=pd.DataFrame()    
for i in range(0,7):
    indx=rf_fea_importances.iloc[i][0]
    print(indx)
    test_new[indx]=test[indx]   

#ADD THE CATERGORICAL VARIABLES BACK
cat_enc=[]
new_cat_enc=[]


t=train.dtypes
cat_enc=pd.get_dummies(t[cat_fea],prefix=cat_fea).astype(int)
train_new=pd.concat([train_new,cat_enc],1)


t=test
cat_enc=pd.get_dummies(t[cat_fea],prefix=cat_fea).astype(int)
test_new=pd.concat([test_new,cat_enc],1)

train_new['meal_id'].unique()
train_new['week'].unique()
train_new['homepage_featured'].value_counts()#ALREADY BINARIZED
test_new['center_id'].unique()
train_new.drop('week',1,inplace=True)
test_new.drop('week',1,inplace=True)


new_cat_fea=['meal_id','center_id']
t=train_new

 

t[new_cat_fea]
t.dtypes
t[new_cat_fea]=t[new_cat_fea].astype(str)
t.dtypes
new_cat_enc=[]

new_cat_enc=pd.get_dummies(t[new_cat_fea]).astype(int)
train_new=pd.concat([train_new,new_cat_enc],1)
train_new.drop(new_cat_fea,inplace=True,axis=1)
test_new=pd.concat([test_new,new_cat_enc],1)
test_new.drop(new_cat_fea,inplace=True,axis=1)



train_new.dtypes

#MISSING VALUES--2
train_new.isnull().sum()

train_new['center_type_TYPE_C'].unique()

for var in train_new.columns:
    r=train_new[var].mode()
    r[0]
    train_new[var].fillna(r[0],inplace=True)

for var in test_new.columns:
    r=test_new[var].mode()
    r[0]
    test_new[var].fillna(r[0],inplace=True)

train_new.isnull().sum()



xtrain=mms.fit_transform(train_new)
xtest=mms.fit_transform(test_new)

x_train=[]
y_train=[]


for i in range(60,len(xtrain)):
    x_train.append(xtrain[i-60:i])
    y_train.append(y_sc[i,0])


import xgboost
from xgboost import XGBRegressor
xgb=XGBRegressor()
xgb.fit(xtrain,y_sc)


def train_lgb(max_depth=5,seed=4,num_round=2500):
    d_train = lgbm.Dataset(X,Y)
    params = {
        'objective' :'regression',
        'max_depth':max_depth,
        'learning_rate' : 0.1,
        'num_leaves' :(2*max_depth)-1 ,
        'feature_fraction': 0.8,
        "min_data_in_leaf" : 100,
        'bagging_fraction': 0.7, 
        'boosting_type' : 'gbdt',
        'metric': 'rmse',
        'seed':seed
    }
    lgb= lgbm.train(params, d_train, num_round)
    return lgb
lgb1=train_lgb(6,4,2500)

ddtest=xgb.DMatrix(test.drop(['id','week','center_id','meal_id'],axis=1))
pred=bst.predict(ddtest)

#y_pred=xgb.predict(xtest)
sub=pd.read_csv('sample_submission.csv')
sub.to_csv('ensemble-xgb-lgb-1.csv',index=False)
sub.head()






