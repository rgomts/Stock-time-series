import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from subprocess import check_output
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from pandas.plotting import lag_plot
import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.dates as mdates

warnings.filterwarnings('ignore')

tick_name = input("ticker name: ")
co_name = input("company name: ")

fp = input("filepath (csv): ")
df = pd.read_csv(fp)
df.head()
print(df.head())
print(df.shape)
print(df.columns)

df[['Close']].plot()
plt.title(co_name)
plt.show()

dr = df.cumsum()
dr.plot()
plt.title(co_name + ' Cumulative Returns')
plt.show()

#plt.figure(figsize=(10,10))
lag_plot(df['Open'], lag=5)
print(lag_plot(df['Open'], lag=5))
plt.title(co_name + 'Autocorrelation plot')
plt.show()



d1_v = df.values
d1 = np.transpose(d1_v)
p = np.array(df['Open'])

dates = np.array(df['Date'])
d = mdates.date2num(dates)

coef_lr1 = np.polyfit(d, p,1)
print("coef:",coef_lr1)
lr1 = np.poly1d(coef_lr1)
print("lr1:",lr1)

coef_lr2 = np.polyfit(d, p,2)
print("coef:",coef_lr2)
lr2 = np.poly1d(coef_lr2)
print("lr2:",lr2)

coef_lr3 = np.polyfit(d, p,3)
print("coef:",coef_lr3)
lr3 = np.poly1d(coef_lr3)
print("lr3:",lr3)

coef_lr6 = np.polyfit(d, p,6)
print("coef:",coef_lr6)
lr6 = np.poly1d(coef_lr6)
print("lr6:",lr6)

coef_lr10 = np.polyfit(d, p,10)
print("coef:",coef_lr10)
lr10 = np.poly1d(coef_lr10)
print("lr10:",lr10)


'''fig, cx = plt.subplots()

xx = np.linspace(x.min(), x.max(), 100)
dd = mdates.num2date(xx)
cx.plot(dd, lr1(xx), '-g')
cx.plot(dates, p, '+', color='b', label='blub')
plt.scatter(dates, p)'''



plt.plot(d,lr1(d),label="lr1")

plt.plot(d,lr2(d),color= 'c', label="lr2")
plt.plot(d,lr3(d),color= 'k', label="lr3")
plt.plot(d,lr6(d),color= 'g', label="lr6")
plt.plot(d,lr10(d),color= 'r', label="lr10")
plt.scatter(d, p)
plt.show()





