#5th step of the project predicting the stock prices for 30days
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import matplotlib.pyplot as plt
#importing csv file to initialise series
data=pd.read_csv("RELIANCE_data.CSV",skiprows=2)
data['Date']=pd.to_datetime(data['Date'],dayfirst=True)
data.set_index('Date',inplace=True)
series=data["Close"]
#print(data.head())
diff_series=series.diff().dropna()
fig, axes = plt.subplots(2,1)
plot_acf(diff_series, lags=40,ax=axes[0])
plot_pacf(diff_series, lags=40,ax=axes[1])
model=ARIMA(series,order=(0,1,0))  #most important line of the code
model_fit=model.fit()
print("-----THIS IS MODEL FIT SUMMARY-----")
print(model_fit.summary())
plt.tight_layout()
plt.show()
forecast=model_fit.forecast(steps=30)
print("-----30 DAYS FORECAST SUMMARY-----")
print(forecast)
future_dates = pd.bdate_range(
    start=series.index[-1] + pd.Timedelta(days=1),
    periods=30
)

plt.figure(figsize=(12,6))
plt.plot(series, label='Actual price')
plt.plot(future_dates, forecast.values, label='30 days Forecast')

plt.title('RELIANCE 30 DAYS STOCK FORECAST')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
# Train/Test Split
train_size = int(len(series) * 0.8)
train = series[:train_size]
test = series[train_size:]

print(f"Train: {len(train)} days, Test: {len(test)} days")

# Model on train data
model2 = ARIMA(train, order=(0,1,0))
model_fit2 = model2.fit()

predictions = model_fit2.forecast(steps=len(test))

# MAE/RMSE
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = mean_absolute_error(test, predictions)
rmse = np.sqrt(mean_squared_error(test, predictions))

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# Actual vs Predicted Graph
plt.figure(figsize=(12,6))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, label='Predicted')
plt.legend()
plt.title('Actual vs Predicted - RELIANCE')
plt.show()