from matplotlib import pyplot
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
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

dataset = pd.read_csv('data/finaldata.csv', header=0)
dataset=dataset[['r_date','date','cnt','day_of_week']]
for i in range(len(dataset)):
	if (dataset.loc[i,'date']>0)==False :
		dataset.loc[i, 'date']=dataset.loc[i-1,'date']
		print(dataset.loc[i, 'date'])
print(dataset.info())
dataset=dataset.groupby(['r_date','date','day_of_week']).agg('sum').reset_index()
# dataset['date']=range(1,len(dataset)+1)
# dataset.cnt=dataset['cnt'].rolling(window=7,min_periods=1).mean()
print(dataset.info())
dataset.columns=['r_date','date','day_of_week','cnt']
dataset=dataset[['date','day_of_week','cnt']]
dataset['cnt']=np.log1p(dataset['cnt'])
print(dataset.info())
values=dataset.values.astype('float32')
print(np.shape(values))
trainsize=int(len(values)*0.65)
print(trainsize)
train = values[:-365, :]
test = values[-365:, :]
t_y = train[7:, -1]
scaler = MinMaxScaler(feature_range=(0, 1))
print(np.mean(train[:,-1]))
scaled = scaler.fit_transform(train)
print(np.mean(scaled[:,-1]))
reframed = series_to_supervised(scaled,7, 1)
# reframed.drop(reframed.columns[[23]], axis=1, inplace=True)

values = reframed.values
test_y=test[:, -1].copy()
test[:, -1]=np.zeros((np.shape(test)[0]))
test=scaler.transform(test)[:, :-1]
# test_X=scaler.transform(test)
train_X, train_y = values[:, :-1], values[:, -1]
# test_X, test_y = test_X[:, :-1], test_X[:, -1]

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
# test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
# print(train_X.shape, train_y.shape,test_X.shape, test_y.shape)

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
history = model.fit(train_X, train_y, epochs=250, batch_size=100, verbose=2, shuffle=False)
# plot history
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()
# pyplot.show()
# make a prediction
y=[]
X=train_X[-1,:,:].reshape(1,train_X.shape[2])
X = np.concatenate((X, train_y[-1].reshape(1, 1)), axis=1)
for i in range(len(test)):
	X=np.concatenate((X[:,3:],test[i].reshape(1,2)), axis=1).reshape((1, 1, train_X.shape[2]))
	# if i%30==0:
	# 	history = model.fit(train_X, train_y, epochs=10, batch_size=100, verbose=2, shuffle=False)
	p=model.predict(X)[0,0]
	# if len(y)>1 and ((p<0.5*y[-1])or(p>2*y[-1])):
	# 	p=0.8*p
	# if len(y)>1 and ( ((y[-1]>2*y[-2])and(p>1.2*y[-1]))):
	# 	p=0.8*p
	y.append(p)
	# train_X = np.concatenate((train_X, X), axis=0)
	# p=np.array([p])
	# train_y = np.concatenate((train_y, p*1.2), axis=0)
	X = np.concatenate((X.reshape(1,train_X.shape[2]), (y[-1]).reshape(1, 1)), axis=1)
print(y)
# yhat = model.predict(test_X)
# print(np.shape(yhat))
# test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# # invert scaling for forecast
inv_y = model.predict(train_X)
print(np.shape(train_X.reshape(train_X.shape[0],train_X.shape[2])),np.shape(inv_y))
inv_y = np.concatenate((train_X.reshape(train_X.shape[0],train_X.shape[2])[:, 0:2],np.array(inv_y)), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,-1]
rmse =mean_squared_error(np.expm1(t_y), np.expm1(inv_y))
print('train RMSE: %.3f' % rmse)
pyplot.plot(np.expm1(inv_y), label='preds')
pyplot.plot(np.expm1(t_y), label='true')
pyplot.legend()
pyplot.show()

inv_yhat = np.concatenate((test[:, 0:2],np.array(y).reshape(test.shape[0],1)), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,-1]
print(np.expm1(np.mean(inv_yhat)))
# test_y=test_y.reshape(test_y.shape[0],1)
# # invert scaling for actual

rmse =mean_squared_error(np.expm1(test_y), np.expm1(inv_yhat))
print('Test RMSE: %.3f' % rmse)
pyplot.plot(np.expm1(inv_yhat), label='preds')
pyplot.plot(np.expm1(test_y), label='true')
pyplot.legend()
pyplot.show()
# print(inv_y)
# calculate RMSE
