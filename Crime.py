"""

Crimes in Boston Time Series Analysis
Author : Gabriel G. Carvalho
Time Series Analysis and Forecast @CInUFPE

"""

######################################## IMPORTING ############################
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import pmdarima as pm
from pykalman import KalmanFilter
from statsmodels.tsa.ar_model import AR
import seaborn as sns
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr 
from statsmodels.tsa.arima_process import ArmaProcess
from arch import arch_model
sns.set()
##########################  PLOTS  ############################################

###############################################################
########## ARMA(3,1) // ARIMA (1,1,1) com Log #################
###############################################################


crime = pd.read_csv("Crime.csv") # Using the closing price of 
								  # the monthly time series.
crime = crime["x"]
plt.plot(crime,linewidth=1,color="k")
plt.title("Monthly Boston armed robberies")
plt.xlabel("Time (Months)")
plt.ylabel("Crimes")
plt.show()

crime = np.log(crime) # Log time series
split = int(0.7*len(crime))
Train, Test = crime[:split] ,crime[split:] # Train and Test series
crime_returns = np.diff(crime)
# print(crime.head())

# # 
plt.plot(crime,c="k",linewidth=1)
plt.xlabel("Time (Months)")
plt.ylabel("Crimes (Log)")
plt.title("Monthly Boston armed robberies")
plt.show()

# BTC - USD monthly returns
plt.plot(crime_returns,c="k",linewidth=1)
plt.xlabel("Time (Months)")
plt.ylabel("Returns")
plt.title("Returns")
plt.show()

# ACF plot
plot_acf(crime, lags=20, c="k")
plt.show()

# PACF plot
plot_pacf(crime, lags=20, c= "k")
plt.show()

# ACF plot
plot_acf(crime_returns, lags=20, c="k")
plt.title("ACF (Returns)")
plt.show()

# PACF plot
plot_pacf(crime_returns, lags=20, c= "k")
plt.title("PACF (Returns)")
plt.show()


################# Holt and Winters ############################################
# fit the data
hw = Holt(Train).fit(smoothing_level=0.7, smoothing_slope=0.08)
# plot the data
hw_fit=hw.fittedvalues
plt.plot(hw_fit,label='HW Fit')
plt.plot(Train,label="Train")
plt.plot(Test,label="Test")
# to show the plot
hw_fc=hw.forecast(len(Test))
plt.plot(hw_fc,label="HW Forecast")
plt.title("Holt-Winters")
plt.legend(loc='best')
plt.show()
# MSE
mse_hw_train= mean_squared_error(Train,hw_fit)
print("MSE HW Train=",mse_hw_train)
mse_hw= mean_squared_error(Test,hw_fc)
print("MSE HW Test=",mse_hw)

# ################# Auto Regressive #############################################
# AR(p) = ARIMA(p,0,0)
model_ar = ARIMA(Train, order=(1,0, 0))  
fitted_ar = model_ar.fit(disp=0)  
# Forecast
fc_ar, se_ar, conf_ar = fitted_ar.forecast(len(Test), alpha=0.05)  # 95% conf

# Make as pandas series
fc_series_ar = pd.Series(fc_ar, index=Test.index)

ar_fittedValues = fitted_ar.fittedvalues

# Plot
fitted_ar.plot_predict(dynamic=False,end=len(Train)+len(Test))
plt.plot(Test,label="Test",color="k")
plt.plot(fc_series_ar, label='Forecast',color="green")
#plt.title('Forecast vs Actuals')
#plt.legend(loc='upper left', fontsize=8)
plt.title("AR")
L=plt.legend(loc='best')
L.get_texts()[0].set_text('AR Fit')
L.get_texts()[1].set_text('Train')
plt.show()
# MSE
mse_ar_train = mean_squared_error(Train, ar_fittedValues)
print("MSE AR Train= ", mse_ar_train)
mse_ar = mean_squared_error(Test, fc_series_ar)
print("MSE AR Test = ", mse_ar)

# ############################# ARMA ##############################################

# ARMA(p,q) = ARIMA(p,0,q)
model_arma = ARIMA(Train, order=(1,0, 1))  
fitted_arma = model_arma.fit(disp=0)  
# Forecast
fc_arma, se_arma, conf_arma = fitted_arma.forecast(len(Test), alpha=0.05)  # 95% conf

# Make as pandas series
fc_series_arma = pd.Series(fc_arma, index=Test.index)
arma_fittedValues = fitted_arma.fittedvalues
# Plot
fitted_arma.plot_predict(dynamic=False,end=len(Train)+len(Test))
plt.plot(Test,label="Test",color="k")
plt.plot(fc_series_arma, label='Forecast',color="green")
#plt.title('Forecast vs Actuals')
#plt.legend(loc='upper left', fontsize=8)
plt.title("ARMA")
L=plt.legend(loc='best')
L.get_texts()[0].set_text('AR Fit')
L.get_texts()[1].set_text('Train')
plt.show()
# MSE
mse_arma_train = mean_squared_error(Train, arma_fittedValues)
print("MSE ARMA Train= ", mse_arma_train)
mse_arma = mean_squared_error(Test, fc_series_arma)
print("MSE ARMA Test= ", mse_arma)


# # ################################## ARIMA ######################################
# model = ARIMA(train, order=(p,d,q))  
model = ARIMA(Train, order=(1, 1, 1))  
fitted = model.fit(disp=0)  
# Forecast
fc, se, conf = fitted.forecast(len(Test), alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=Test.index)
arima_fittedValues = fitted.fittedvalues
# Plot
fitted.plot_predict(dynamic=False,start=1,end=len(Train)+len(Test))
plt.plot(Test,label="Test",color="k")
plt.plot(fc_series, label='Forecast',color="green")
#plt.title('Forecast vs Actuals')
#plt.legend(loc='upper left', fontsize=8)
plt.title("ARIMA")
L=plt.legend(loc='best')
L.get_texts()[0].set_text('ARIMA Fit')
L.get_texts()[1].set_text('Train')
plt.show()
# MSE
mse_arima_train = mean_squared_error(crime_returns[1:split], arima_fittedValues)
print("MSE ARIMA Train (on returns)= ", mse_arima_train)
mse_arima = mean_squared_error(Test, fc_series)
print("MSE ARIMA Test= ", mse_arima)


# print(fitted.summary())
#plot residual errors
residuals = pd.DataFrame(fitted.resid)
# plt.plot(residuals)
# plt.show()
plot_acf(residuals,color="k")
plt.title("ACF ARIMA(1,1,1) residuals")
plt.show()
plot_pacf(residuals,color="k")
plt.title("PACF ARIMA(1,1,1) residuals")
plt.show()
# residuals.plot(kind='kde')
# plt.show()
# print(residuals.describe())


# ########################### KALMAN FILTER #######################################

kf = KalmanFilter(transition_matrices = [1],
                  observation_matrices = [1],
                  initial_state_mean = 3.713572,
                  initial_state_covariance = 1,
                  observation_covariance=1,
                  transition_covariance=0.85)

# Use the observed values of the price to get a rolling mean
state_means, _ = kf.filter(crime.values)
state_means = pd.Series(state_means.flatten(), index=crime.index)

# Compute the rolling mean with various lookback windows
mean6 = pd.Series(crime).rolling(window=6).mean()

mean15 = pd.Series(crime).rolling(window=15).mean()


# Plot original data and estimated mean
plt.plot(state_means[split:],label="Kalman Estimate")
plt.plot(crime[split:],label="Actual",color="k",linewidth=1)
# plt.plot(mean6,label="6W moving average")
# plt.plot(mean15,label="15W moving average")
plt.title('Kalman filter estimate of average')
plt.legend()
plt.xlabel('Months')
plt.ylabel('Log of Prices')
plt.show()
mse_kalman = mean_squared_error(crime, state_means)
print("MSE KALMAN = ", mse_kalman)

# # # model = pm.auto_arima(btc, start_p=0, start_q=0,
# # #                       test='adf',       # use adftest to find optimal 'd'
# # #                       max_p=10, max_q=10, # maximum p and q
# # #                       m=1,              # frequency of series
# # #                       d=None,           # let model determine 'd'
# # #                       seasonal=False,   # No Seasonality
# # #                       start_P=0, 
# # #                       D=0, 
# # #                       trace=True,
# # #                       error_action='ignore',  
# # #                       suppress_warnings=True, 
# # #                       stepwise=True)

# # # print(model.summary())


