
# LOADING AND HANDLING TIME SERIES DATA IN PANDAS
import pandas as pd
import numpy as np 
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=15,6

data=pd.read_csv("/home/subir_sbr/Downloads/AirPassengers.csv")
print(data.head())
print('\nData Types: ')
print(data.dtypes)

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
data = pd.read_csv('/home/subir_sbr/Downloads/AirPassengers.csv', parse_dates=['Month'], index_col='Month',date_parser=dateparse)
print(data.head())
print('\nData Types: ')
print(data.dtypes)
# parse dates: this specifies the column which contains the date-time 
# information. As we say above the the column name is 'Month'

#index_col: A key idea behind using pandas TS data is that the index has to be
# the variables depicting date-time information. Sothis tells pandas to use the 'Month'
# column as index

#This specifies a function which converts an input string into datetime variable. 
#Be default Pandas reads data in format ‘YYYY-MM-DD HH:MM:SS’. If the data is not in this format, 
#the format has to be manually defined. Something similar to the dataparse 
#function defined here can be used for this purpose.

print(data.index)
ts = data['#Passengers']
print(ts.head(10))
from datetime import datetime

print(ts[:datetime(1949,5,1)])
print(ts['1949'])

#########################################
# CHECK STATIONARITY OF A TIME SERIES

# TS is said to be stationary if its statistical properties such as
# mean,variance remain constant over time.
# and an autocovariance that does not depend on time.
#plt.plot(ts,color='green',label='Timeseries plot')
#plt.show()

# Might not always possible to check stationarity with visuals.
#check stationarity with following rules:
# (1) Plotting rolling statistics
# (2) Dickey-Fuller Test

# Please note that I’ve plotted standard deviation instead of variance to keep the unit similar to mean.

from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):
	# determine rolling statistics
	rolmean=pd.Series.rolling(timeseries,center=False,window=12).mean()
	rolstd=pd.Series.rolling(timeseries,center=False,window=12).std()

	#plot rolling statistics:
	orig=plt.plot(timeseries,color='blue',label='Original')
	mean=plt.plot(rolmean,color='red',label='Rolling mean')
	std=plt.plot(rolstd,color='black',label='Rolling std')
	plt.legend(loc='best')
	plt.title('Rolling Mean & Standard Deviation')
	plt.show(block=True)

	# Dickey-Fuller test:
	print("Result of Dickey-Fuller test")
	dftest=adfuller(timeseries,autolag='AIC')
	dfoutput= pd.Series(dftest[0:4],index=['Test Statistic','P-value','# Lags used','NUmber of Observations Used'])
	for key,value in dftest[4].items():
		dfoutput['Critical Value (%s)' % key]=value
	print(dfoutput)



#test_stationarity(ts)
# How to make time series stationary
# Non-stationarity is due to 1) Trend and 2) Seasonality

# Estimating and Eliminating Trend
import numpy as np
ts_log=np.log(ts) #apply transformation which penalize higher values more than smaller values.
#Moving average
moving_avg=pd.Series.rolling(ts_log,window=12).mean()
#plt.plot(ts_log)
#plt.plot(moving_avg,color='red')


ts_log_moving_avg_diff=ts_log-moving_avg
print(ts_log_moving_avg_diff.head(12))
ts_log_moving_avg_diff.dropna(inplace=True)
'''test_stationarity(ts_log_moving_avg_diff)'''


#the test statistic is smaller than the 5% critical values so we can say
# with 95% confidence that this is a stationary series.

# a ‘weighted moving average’ where more recent values are given a higher weight. 
#There can be many technique for assigning weights. A popular one is exponentially
# weighted moving average where weights are assigned to all the previous values 
#with a decay factor.

expweighted_avg=pd.Series.ewm(ts_log,halflife=12).mean()
#Series.ewm(halflife=12,ignore_na=False,min_periods=0,adjust=True).mean()

#plt.plot(ts_log)
#plt.plot(expweighted_avg,color='red')
ts_log_ewma_diff=ts_log-expweighted_avg
#test_stationarity(ts_log_ewma_diff)

######### ELIMINATING TREND AND SEASONALITY ###########
'''
we have two ways of removing trend:
(1) Differencing: taking a difference with a particular time lag
(2) Decomposition:modeling both trand and seasonality and removing them
	from model.

'''
ts_log_diff = ts_log - (ts_log.shift()).shift()
#plt.plot(ts_log_diff)
#plt.show()
ts_log_diff.dropna(inplace=True)
#test_stationarity(ts_log_diff)

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition=seasonal_decompose(ts_log)

trend=decomposition.trend
seasonal=decomposition.seasonal
residual=decomposition.resid
'''
plt.subplot(411)
plt.plot(ts_log,label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend,label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual,label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
'''

ts_log_decompose=residual
ts_log_decompose.dropna(inplace=True)
test_stationarity(ts_log_decompose)




