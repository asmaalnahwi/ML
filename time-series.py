# -*- coding: utf-8 -*-
"""
Created on Sun Jul 08 08:53:01 2018

@author: AAlNahwi
"""

from datetime import datetime 
from dateutil import parser 
import numpy as np
import pandas as pd
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt 
import seaborn; seaborn.set()
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import *
from statsmodels.graphics import tsaplots
# Import statsmodel
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.ar_model import AR

ds = pd.read_csv('google.csv')
#==============================================================================
# datetime(year=2018, month=4, day=27)
# date=parser.parse("4th of july,2015")
# print date 
# 
# date=np.array('2015-07-04', dtype=np.datetime64)
# print date
# np.datetime64('2015-07-04 12:00')
# date=pd.to_datetime("4th of july,2015")
# index=pd.DatetimeIndex(['2014-07-01','2014-07-02','2019-07-07'])
# data=pd.Series([0,1,2],index=index)
# print data
# print data['2014-07-02':'2014-07-07']
# print data['2014']
#==============================================================================
#==============================================================================
#'''Export the data '''
#ds.date = pd.DatetimeIndex(ds.date)
ds['date']=pd.to_datetime(ds['date'])
#index=pd.PeriodIndex(ds.date, freq='M')
index=ds['date']
c=pd.DataFrame(data=ds['close'])
c.index=index
c_log = np.log(c)    #can help to stabilize the variance of a time series.
#c_log.plot(kind = "hist", bins = 30)
Results = pd.DataFrame(columns = ["Model", "Forecast", "RMSE"])

###############################################################################
#''' stationarity test           '''

from statsmodels.tsa.stattools import adfuller


def test_stationarity(timeseries):

    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(index[13:],timeseries.close, color='blue',label='Original')
    mean = plt.plot(index[13:],rolmean, color='red', label='Rolling Mean')
    std = plt.plot(index[13:],rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
   #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries.close, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput

################################################################################
#''' way to stationarize the data '''

#==============================================================================
#''' first diffrecne '''
fDiff=c - c.shift(1)

#==============================================================================
#''' seseonal diffrecne '''
sDiff=c - c.shift(12)

#=============================================================================
#'''the seasonal first difference values '''
dsDiff=fDiff-fDiff.shift(12)
dsDiff = dsDiff[dsDiff.close.notnull()]
#=============================================================================
##############################################################################################################
def RMSE(predicted, actual):
    mse = (predicted - actual)**2
    rmse = np.sqrt(mse.sum()/len(mse))
    return rmse










###############################################################################################################

################################################################################################################
#'''Plot the ACF and PACF charts and find the optimal parameters'''
#fig = plt.figure(figsize=(12,8))
#ax1 = fig.add_subplot(211)
#fig =tsaplots.plot_acf(dsDiff[13:], lags=40, ax=ax1)
#ax2 = fig.add_subplot(212)
#fig =tsaplots.plot_pacf(dsDiff[13:], lags=40, ax=ax2)

################################################################################################################
def result(model, forcast, RMSE):
   
    i=len(Results)
    Results.loc[i,"Model"] = model
    Results.loc[i,"Forecast"] = forcast
    Results.loc[i,"RMSE"] = RMSE
    Results.head()





#################################################################################################################

#''' build the model '''

#'''     Mean model   '''

#model_mean_pred=ds.close.mean()
#ds['closemean']=model_mean_pred
#ds.plot(kind="line", x="date", y = ["close", "closemean"])
#=============================================================================

#''' linear model '''
#ds["timeIndex"] = ds.date - ds.date.min()
#ds["timeIndex"] =  ds["timeIndex"]/np.timedelta64(1, 'M')
#ds["timeIndex"] = ds["timeIndex"].round(0).astype(int)
#f = 'close ~ timeIndex'
#y, x = patsy.dmatrices(f, ds, return_type='dataframe')
##model_linear = smf.ols(formula='close ~ timeIndex', data = ds).fit()








###############################################################################################################

#''' AR model '''
X = c_log.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
# train autoregression
model = AR(train)
model_fit = model.fit()
print('Lag: %s' % model_fit.k_ar)
print('Coefficients: %s' % model_fit.params)
# make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
for i in range(len(predictions)):
	print('predicted=%f, expected=%f' % (predictions[i], test[i]))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot results
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()


















#########################################################################################################
#''' ARIMA model '''
#==============================================================================
## fit model
#model = ARIMA(c, order=(15,1,0))
#model_fit = model.fit(disp=0)
#print(model_fit.summary())
## plot residual errors
#residuals = pd.DataFrame(model_fit.resid)
#residuals.plot()
##pyplot.show()
#residuals.plot(kind='kde')
##pyplot.show()
#print(residuals.describe())
#autocorrelation_plot(c)
#plt.show()


#==============================================================================

X=c_log.values
size = int(len(X) * 0.80)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(15,1,12))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	y= output[0]
	predictions.append(y)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (y, obs))
error = mean_squared_error(test, predictions)

print('Test MSE: %.3f' % error)

# plot
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()
#==============================================================================
###############################################################################################################
#''' moving average '''
#c_log = np.log(c)
##c_log.plot()
#
#moving_avg = pd.rolling_mean(c_log ,12)
#fig=plt.figure()
#ax=fig.add_axes([0.1,0.1,0.8,0.8]) 
#ax.plot(index,c_log.close, c = 'red', label = 'orginal') 
#ax.plot(index,moving_avg.close, c = 'green', label = 'Average')
#ax.legend()
#
#error = mean_squared_error(c_log.close, moving_avg.close)
#
#print('Test MSE: %.3f' % error)
##c_log_moving_avg_diff = c_log - moving_avg
#
#c_log_moving_avg_diff.dropna(inplace=True)
#test_stationarity(c_log_moving_avg_diff)

























#plt.plot(np.arange(1080),ds['close'].astype(float))
#autocorrelation_plot(c)
#c.plot(kind = "hist", bins = 30)
#c.plot()
#plt.show()