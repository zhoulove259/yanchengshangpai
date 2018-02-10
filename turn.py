import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


data=pd.read_csv('data/data.csv')
data.drop('date',inplace=True,axis=1)
train=data[data['cnt']>0]
low=train[train['cnt']<80]
print(len(low))
print(np.mean(train['cnt']))
for i in range(5):
    train=pd.concat([low.copy(),train],axis=0)
y=train['cnt']
miny=min(y)
maxy=max(y)
y = (y-miny)/(maxy-miny)
X=train.drop(['cnt','r_date'],axis=1)
print(np.shape(X))
test=data[~(data['cnt']>0)]
test=test.drop(['cnt','r_date'],axis=1)
print(np.shape(test))
params = {'booster': 'gbtree',
          'objective': 'reg:logistic',
          'eval_metric': 'rmse',
          'gamma': 0.1,
          'min_child_weight': 2,
          'max_depth': 5,
          'lambda': 10,
          'alpha': 2.5,
          'silent':True,
          'subsample ':0.8,
          'colsample_bytree': 0.5,
          'colsample_bylevel': 0.5,
          'eta': 0.01,
          'tree_method': 'gpu_exact',
          'seed': 50,
          'gpu_id': 0,
          # 'scale_pos_weight':10,
          'nthread': -1
          }
def loss(preds,dtrain): #preds是结果（概率值），dtrain是个带label的DMatrix
    labels=dtrain.get_label() #提取label
    cha = (preds - labels)*(maxy-miny)
    trainmse = np.dot(cha, cha.T) / len(cha)
    return 'mse',trainmse
xgb1 = XGBRegressor(
 learning_rate =0.01,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=2,
 gamma=0.1,
 reg_alpha=2.5,
 reg_lambda=5,
 subsample=0.8,
 colsample_bytree=0.5,
 objective= 'reg:logistic',
 nthread=-1,
 scale_pos_weight=1,
 silent=True,
 tree_method= 'gpu_exact',
 gpu_id= 0,
 seed=0)
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2, random_state=0)
result=xgb.cv(params, xgb.DMatrix(X,label=y), num_boost_round=1000, nfold=8, stratified=False, folds=None,maximize=False, early_stopping_rounds=10,as_pandas=True, verbose_eval=None, show_stdv=True,
       seed=0, callbacks=None, shuffle=True,feval=loss)
xgb1.set_params(n_estimators=result.shape[0])
xgb1.fit(X_train, Y_train)


preds=xgb1.predict(X_train)*(maxy-miny)+miny
# preds=preds+(preds-np.mean(preds))*0.5
cha=(preds - Y_train*(maxy-miny)-miny)
print('train',np.dot(cha,cha.T)/len(cha))


preds=xgb1.predict(X_test)*(maxy-miny)+miny
# preds=preds+(preds-np.mean(preds))*0.5
cha=(preds - Y_test*(maxy-miny)-miny)
print('test',np.dot(cha,cha.T)/len(cha))
print(np.min(preds),np.max(preds),np.mean(preds))


xgb1.fit(X,y)
preds=xgb1.predict(test)*(maxy-miny)+miny
data=pd.read_csv('data/data.csv')
data.loc[~(data['cnt']>0),'cnt']=preds
print(data.info())
data.to_csv('data/finaldata.csv',index=None)
print(np.mean(train['cnt']),np.mean(preds))


