import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
from subprocess import check_output
import seaborn as sns
import warnings
from pandas.plotting import lag_plot
import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.dates as mdates
#import fastai
#from fastai.tabular import add_datepart
#from structured import  add_datepart
from sklearn.linear_model import LinearRegression
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
#from pyramid.arima import auto_arima
from fbprophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

warnings.filterwarnings('ignore')

tick_name = input("ticker name: ")
co_name = input("company name: ")

fp =input("filepath (csv): ")
df = pd.read_csv(fp)
df.head()
print(df.head())
print(df.shape)
print(df.columns)

df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']

data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])

for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

train = new_data[:987]
valid = new_data[987:]

# shapes of training set
print('\n Shape of training set:')
print(train.shape)

# shapes of validation set
print('\n Shape of validation set:')
print(valid.shape)

# In the next step, we will create predictions for the validation set and check the RMSE using the actual values.
# making predictions
preds = []
for i in range(0,valid.shape[0]):
    a = train['Close'][len(train)-248+i:].sum() + sum(preds)
    b = a/248
    preds.append(b)

# checking the results (RMSE value)
rms=np.sqrt(np.mean(np.power((np.array(valid['Close'])-preds),2)))
print('\n RMSE value on validation set:')
print(rms)

df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']

#sorting
data = df.sort_index(ascending=True, axis=0)

#creating a separate dataset
'''new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])

for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]'''


#some statistical infrences

#linear reg

d1_v = df.values
d1 = np.transpose(d1_v)
o = np.array(df['Open'])
c = np.array(df['Close'])
dates = np.array(df['Date'])
d = mdates.date2num(dates)

o_coef_lr1 = np.polyfit(d, o,1)
print("coef:",o_coef_lr1)
o_lr1 = np.poly1d(o_coef_lr1)
print("lr1 Open:",o_lr1)

o_coef_lr2 = np.polyfit(d, o,2)
print("coef:",o_coef_lr2)
o_lr2 = np.poly1d(o_coef_lr2)
print("lr2 Open:",o_lr2)

o_coef_lr3 = np.polyfit(d, o,3)
print("coef:",o_coef_lr1)
o_lr3 = np.poly1d(o_coef_lr3)
print("lr3 Open:",o_lr3)

o_coef_lr5 = np.polyfit(d, o,5)
print("coef:",o_coef_lr5)
o_lr5 = np.poly1d(o_coef_lr5)
print("lr5 Open:",o_lr5)

o_coef_lr10 = np.polyfit(d, o,10)
print("coef:",o_coef_lr10)
o_lr10 = np.poly1d(o_coef_lr10)
print("lr1 Open:",o_lr10)

plt.plot(d,o_lr1(d), label="lr1")
plt.plot(d,o_lr2(d),color= 'c', label="lr2")
plt.plot(d,o_lr3(d),color= 'k', label="lr3")
plt.plot(d,o_lr5(d),color= 'g', label="lr5")
plt.plot(d,o_lr10(d),color= 'r', label="lr10")
plt.scatter(d, o )
plt.show()

c_coef_lr1 = np.polyfit(d, c,1)
print("coef:",c_coef_lr1)
c_lr1 = np.poly1d(c_coef_lr1)
print("lr1 close:",c_lr1)

c_coef_lr2 = np.polyfit(d, c,2)
print("coef:",c_coef_lr2)
c_lr2 = np.poly1d(c_coef_lr2)
print("lr2 Open:",c_lr2)

c_coef_lr3 = np.polyfit(d, c,3)
print("coef:",c_coef_lr1)
c_lr3 = np.poly1d(c_coef_lr3)
print("lr3 Open:",c_lr3)

c_coef_lr5 = np.polyfit(d, c,5)
print("coef:",c_coef_lr5)
c_lr5 = np.poly1d(c_coef_lr5)
print("lr5 Open:",c_lr5)

c_coef_lr10 = np.polyfit(d, c,10)
print("coef:",c_coef_lr10)
c_lr10 = np.poly1d(c_coef_lr10)
print("lr1 Open:",c_lr10)

plt.plot(d,c_lr1(d), label="lr1")
plt.plot(d,c_lr2(d),color= 'c', label="lr2")
plt.plot(d,c_lr3(d),color= 'k', label="lr3")
plt.plot(d,c_lr5(d),color= 'g', label="lr5")
plt.plot(d,c_lr10(d),color= 'r', label="lr10")
plt.scatter(d, c)
plt.show()
'''df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']

#sorting
data = df.sort_index(ascending=True, axis=0)

#creating a separate dataset
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])

for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]


add_datepart(new_data, 'Date')
new_data.drop('Elapsed', axis=1, inplace=True)  #elapsed will be the time stamp

new_data['mon_fri'] = 0
for i in range(0,len(new_data)):
    if (new_data['Dayofweek'][i] == 0 or new_data['Dayofweek'][i] == 4):
        new_data['mon_fri'][i] = 1
    else:
        new_data['mon_fri'][i] = 0

#split into train and validation
train = new_data[:987]
valid = new_data[987:]

x_train = train.drop('Close', axis=1)
y_train = train['Close']
x_valid = valid.drop('Close', axis=1)
y_valid = valid['Close']

#implement linear regression

model = LinearRegression()
model.fit(x_train,y_train)

preds = model.predict(x_valid)
lr_rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)),2)))
print(lr_rms)

valid['Predictions'] = 0
valid['Predictions'] = preds

valid.index = new_data[987:].index
train.index = new_data[:987].index

plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])'''


#KNN
#split into train and validation
#d1_v = df.values
#d1 = np.transpose(d1_v)
#c = np.array(df['Close'])

df_v = df.values
dv = np.transpose(df_v)
c = np.array(df['Close'])
d = np.array(df['Date'])

'''print(new_data)
#new_data.drop('Elapsed', axis=1, inplace=True)  #elapsed will be the time stamp

new_data['mon_fri'] = 0
for i in range(0,len(new_data)):
    if (new_data['Dayofweek'][i] == 0 or new_data['Dayofweek'][i] == 4):
        new_data['mon_fri'][i] = 1
    else:
        new_data['mon_fri'][i] = 0

#split into train and validation
train = new_data[:987]
valid = new_data[987:]'''

train = df[0:int(len(df)*0.8)]
valid =  df[int(len(df)*0.8):]

x_train = train['Date'].values
y_train = train['Close'].values
x_valid = valid['Date'].values
y_valid = valid['Close'].values


a =x_train.reshape(-1, 1)
b =x_valid.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))

#scaling data
x_train_scaled = scaler.fit_transform(a)
x_train2 = pd.DataFrame(x_train_scaled)
print(x_train2)
x_valid_scaled = scaler.fit_transform(b)
x_valid2 = pd.DataFrame(x_valid_scaled)

#using gridsearch to find the best parameter
params = {'n_neighbors':[2,3,4,5,6,7,8,9]}
knn = neighbors.KNeighborsRegressor()
model = GridSearchCV(knn, params, cv=5)

#fit the model and make predictions
print(y_train)
model.fit(x_train2, pd.DataFrame(y_train.reshape(-1,1)))
preds = model.predict(x_valid2)

#rmse
knn_rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)),2)))
print("knn_rms:",  knn_rms)
#plot
valid['Predictions'] = 0
valid['Predictions'] = preds
plt.plot(valid[['Close', 'Predictions']])
plt.plot(train['Close'])
plt.show()

#arima



'''data = df.sort_index(ascending=True, axis=0)

train = data[:987]
valid = data[987:]

training = train['Close']
validation = valid['Close']

model = auto_arima(training, start_p=1, start_q=1,max_p=3, max_q=3, m=12,start_P=0, seasonal=True,d=1, D=1, trace=True,error_action='ignore',suppress_warnings=True)
model.fit(training)

forecast = model.predict(n_periods=248)
forecast = pd.DataFrame(forecast,index = valid.index,columns=['Prediction'])

arima_rms=np.sqrt(np.mean(np.power((np.array(valid['Close'])-np.array(forecast['Prediction'])),2)))
print(arima_rms)

#plot
plt.plot(train['Close'])
plt.plot(valid['Close'])
plt.plot(forecast['Prediction'])'''

#prophet

#importing prophet


#creating dataframe
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])

for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

new_data['Date'] = pd.to_datetime(new_data.Date,format='%Y-%m-%d')
new_data.index = new_data['Date']

#preparing data
new_data.rename(columns={'Close': 'y', 'Date': 'ds'}, inplace=True)

#train and validation
train = new_data[:987]
valid = new_data[987:]

#fit the model
model = Prophet()
model.fit(train)

#predictions
close_prices = model.make_future_dataframe(periods=len(valid))
forecast = model.predict(close_prices)

#rmse
forecast_valid = forecast['yhat'][987:]
pr_rms=np.sqrt(np.mean(np.power((np.array(valid['y'])-np.array(forecast_valid)),2)))
print("pr_rms ", pr_rms)

#plot
valid['Predictions'] = 0
valid['Predictions'] = forecast_valid.values

plt.plot(train['y'])
plt.plot(valid[['y', 'Predictions']])
plt.show()

#LSTM


#creating dataframe
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

#setting index
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

#creating train and test sets
dataset = new_data.values

train = dataset[0:987,:]
valid = dataset[987:,:]

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

#predicting 246 values, using past 60 from the train data
inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)
lstm_rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))
print("lstm_rms", lstm_rms)

#for plotting
train = new_data[:987]
valid = new_data[987:]
valid['Predictions'] = closing_price
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.show()
