import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# print('turn predict data...')
# test_A = pd.read_table('./data/test_A_20171225.txt')
# sample_A = pd.read_table('./data/sample_A_20171225.txt',header=None)
# nowtime=datetime.date(2015,3,4)
# # print(starttime+datetime.timedelta(days=32))
# timelist=[]
# l=np.shape(test_A)[0]-1
# i=0
# while i<l:
#     d=[nowtime,test_A.loc[i,'date']]
#     timelist.append(d)
#     while i<l and test_A.loc[i,'day_of_week']<=test_A.loc[i+1,'day_of_week']:
#         if test_A.loc[i,'day_of_week']<test_A.loc[i+1,'day_of_week']:
#             nowtime=nowtime + datetime.timedelta(days=1)
#             d = [nowtime, test_A.loc[i+1, 'date']]
#             timelist.append(d)
#         i = i + 1
#     md=test_A.loc[i,'day_of_week']
#     while md<7:
#         nowtime = nowtime + datetime.timedelta(days=1)
#         d = [nowtime, np.nan]
#         timelist.append(d)
#         md=md+1
#     i=i+1
#     if i>=l:
#         break
#     md=test_A.loc[i,'day_of_week']
#     k=1
#     while k<md:
#         nowtime = nowtime + datetime.timedelta(days=1)
#         d = [nowtime,np.nan]
#         timelist.append(d)
#         k=k+1
#     nowtime = nowtime + datetime.timedelta(days=1)
# timelist=timelist[:-3]
# r_timelist=[]
# for t in timelist:
#      r_timelist.append(t)
# data=pd.DataFrame(r_timelist)
# data.columns=['r_date','date']
#
# test_A=pd.merge(data,test_A,how='right',on=['date'])
# data=pd.merge(data,test_A,how='left',on=['r_date','date'])
# week=data.groupby('r_date').day_of_week.median()
# weeks=pd.DataFrame()
# weeks['r_date']=week.index.tolist()
# weeks['day_of_week']=week.values
# data.drop('day_of_week',axis=1,inplace=True)
# data=pd.merge(data,weeks,how='left',on=['r_date'])
# for i in range(len(data)):
#     if(~(data.loc[i,'day_of_week']>0)):
#         if(data.loc[i,'r_date']==data.loc[i-1,'r_date']):
#             data.loc[i, 'day_of_week']= data.loc[i-1, 'day_of_week']
#         else:
#             if(data.loc[i-1,'day_of_week']==7):
#                 data.loc[i , 'day_of_week'] =1
#             else:
#                 data.loc[i, 'day_of_week'] = data.loc[i-1, 'day_of_week']+1
# predata=data

from matplotlib import pyplot
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Activation
import random


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


def parseTest(values):
	weekday=4
	result=[]
	for i in range(len(values)):
		while weekday!=values[i,1]:
			result.append([values[i,0]-0.5,weekday])
			weekday=weekday+1
			if weekday>7:
				weekday=1
		result.append([values[i, 0], weekday])
		weekday = weekday + 1
		if weekday > 7:
			weekday = 1
	print(result)
	return result
dataset = pd.read_csv('data/finaldata.csv', header=0)
dataset=dataset[['r_date','date','cnt','day_of_week']]
for i in range(len(dataset)):
	if (dataset.loc[i,'date']>0)==False :
		dataset.loc[i, 'date']=dataset.loc[i-1,'date']
dataset=dataset.groupby(['r_date','date','day_of_week']).agg('sum').reset_index()
dataset.columns=['r_date','date','day_of_week','cnt']
dataset=dataset[['date','day_of_week','cnt']]
dataset['cnt']=np.log1p(dataset['cnt'])
t_y=dataset['cnt']
print(dataset.info())
values=dataset.values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(dataset.values)
reframed = series_to_supervised(scaled, 7, 1)
# reframed.drop(reframed.columns[[2,5,8,11,14,17,20,21]], axis=1, inplace=True)
values = reframed.values
# values=values[:-1,:]
trainsize=len(values)
test_A = pd.read_table('./data/test_A_20171225.txt')
test_A=np.array(parseTest(test_A.values))
test_A=test_A[1:,:]
# test_A=test_A.values
test_A_back=test_A
sample_A = pd.read_table('./data/sample_A_20171225.txt',header=None)
train = values
test = test_A
train_X, train_y = train[:, :-1], train[:, -1]
print(np.shape(test))
test=np.concatenate((test,np.zeros((np.shape(test)[0],1))), axis=1)
test=scaler.transform(test)[:, :-1]
test_X=test

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape)

from numpy.random import seed
seed(2)
from tensorflow import set_random_seed
set_random_seed(2)

model = Sequential()
model.add(LSTM(32, input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(64,return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(32))
model.add(Dense(1))
model.add(Activation("tanh"))
model.compile(loss='mse', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=200, batch_size=100, verbose=2, shuffle=False)
# make a prediction
y=[]
X=train_X[-1,:,:].reshape(1,train_X.shape[2])
X = np.concatenate((X, train_y[-1].reshape(1, 1)), axis=1)
for i in range(len(test)):
	X=np.concatenate((X[:,3:],test[i].reshape(1,2)), axis=1).reshape((1, 1, train_X.shape[2]))
	y.append(model.predict(X)[0,0] )
	X = np.concatenate((X.reshape(1,train_X.shape[2]), (y[-1]).reshape(1, 1)), axis=1)
print(y)
# yhat = model.predict(test_X)
# print(np.shape(yhat))
# test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# # invert scaling for forecast
inv_yhat = np.concatenate((test[:, 0:2],np.array(y).reshape(test.shape[0],1)), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,-1]
inv_yhat=np.expm1(inv_yhat)
print(inv_yhat)
test_A_back=pd.DataFrame(test_A_back)
test_A_back.columns=['date','value']
test_A_back['value']=inv_yhat
test_A_back.to_csv('result.csv',index=None)

test_A_back=pd.DataFrame(test_A_back)
test_A_back.columns=['date','day_of_week']
test_A_back['value']=inv_yhat
test_A_back=test_A_back[['date','value']]
sample_A.columns=['date','value']
sample_A=sample_A[['date']]
sample_A=pd.merge(sample_A,test_A_back,on='date',how='inner')
sample_A.to_csv('result.csv',index=None)

pyplot.plot(inv_yhat, label='preds')
# pyplot.plot(np.concatenate((t_y.values,inv_yhat), axis=0), label='preds')
pyplot.legend()
pyplot.show()
