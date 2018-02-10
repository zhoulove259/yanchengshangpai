from matplotlib import pyplot
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error



dataset = pd.read_csv('data/finaldata.csv', header=0)
dataset=dataset[['r_date','date','cnt','day_of_week']]
for i in range(len(dataset)):
	if (dataset.loc[i,'date']>0)==False :
		dataset.loc[i, 'date']=dataset.loc[i-1,'date']
		print(dataset.loc[i, 'date'])
dataset=dataset.groupby(['r_date','date','day_of_week']).agg('sum').reset_index()
dataset['date']=range(1,len(dataset)+1)
dataset.columns=['r_date','date','day_of_week','cnt']
dataset=dataset[['date','day_of_week','cnt']]
values=dataset.values.astype('float32')

trainsize=int(len(values)*0.65)
print(trainsize)
train = values[:-365, :]
test = values[-365:, :]
test_y=test[:, -1].copy()
test=test[:, :-1]
# test_X=scaler.transform(test)
train_X, train_y = values[:, :-1], values[:, -1]
# test_X, test_y = test_X[:, :-1], test_X[:, -1]

scaler = MinMaxScaler(feature_range=(0, 1))
train_y = scaler.fit_transform(train_y)

train_X = train_X.reshape((train_X.shape[0], train_X.shape[1]))
params = {'booster': 'gbtree',
          'objective': 'reg:logistic',
          'eval_metric': 'rmse',
          'gamma': 0.15,
          'min_child_weight': 1,
          'max_depth': 4,
          'lambda': 10,
          'alpha': 2.5,
          'silent':True,
          'subsample ':1,
          'colsample_bytree': 0.1,
          'colsample_bylevel': 0.1,
          'eta': 0.01,
          'tree_method': 'gpu_exact',
          'seed': 50,
          'gpu_id': 0,
          # 'scale_pos_weight':10,
          'nthread': -1
          }


model = XGBRegressor(
 learning_rate =0.01,
 n_estimators=1000,
 max_depth=4,
 min_child_weight=1,
 gamma=0.15,
 reg_alpha=2.5,
 reg_lambda=10,
 subsample=1,
 colsample_bytree=0.1,
 colsample_bylevel=0.1,
 objective= 'reg:logistic',
 nthread=-1,
 scale_pos_weight=1,
 tree_method= 'gpu_exact',
 gpu_id= 0,
 seed=0)
print('start cv')
result=xgb.cv(params, xgb.DMatrix(train_X,label=train_y), num_boost_round=1000, nfold=8, stratified=False,  maximize=False, early_stopping_rounds=10,as_pandas=True, verbose_eval=None, show_stdv=True,
       seed=0, callbacks=None, shuffle=True)
model.set_params(n_estimators=int(result.shape[0]))
model.fit(train_X, train_y)
print('start predict')
y=[]
X=train_X[-1,:].reshape(1,train_X.shape[1])
X = np.concatenate((X, train_y[-1].reshape(1, 1)), axis=1)
for i in range(len(test)):
	X=np.concatenate((X[:,3:],test[i].reshape(1,2)), axis=1).reshape((1, train_X.shape[1]))
	y.append(model.predict(X)[0]*(1+i/len(test)))
	X = np.concatenate((X.reshape(1,train_X.shape[1]), y[-1].reshape(1, 1)), axis=1)
print(y)
# yhat = model.predict(test_X)
# print(np.shape(yhat))
# test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# # invert scaling for forecast
inv_yhat = scaler.inverse_transform(y)
# test_y=test_y.reshape(test_y.shape[0],1)
# # invert scaling for actual

rmse =mean_squared_error(test_y, inv_yhat)
print('Test RMSE: %.3f' % rmse)
pyplot.plot(inv_yhat, label='preds')
pyplot.plot(test_y, label='true')
pyplot.legend()
pyplot.show()
# print(inv_y)
# calculate RMSE
