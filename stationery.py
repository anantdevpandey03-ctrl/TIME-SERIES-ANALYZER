import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
df=pd.read_csv("RELIANCE_data.CSV",skiprows=2)
df["Date"]=pd.to_datetime(df["Date"],dayfirst=True)  #date is being converted to index
df.set_index("Date",inplace=True)
df=df[["Close"]].dropna()
plt.figure(figsize=(10,4))
df["Close"].plot(title="Closing Price")


#LOG TRANSFORATION
plt.figure(figsize=(10,4))
df["Log_Close"] = np.log(df["Close"])
df["Log_Close"].plot(title="Log Closing Price")

#DIFFRENCING
plt.figure(figsize=(10,4))
df["Diff_Log_Close"]=df["Log_Close"].diff()
df["Diff_Log_Close"].plot(title="Difference between Closing Price and Log Closing Price")

#ADF (AUGUMENTED DICKEY FULLER TEST)
result=adfuller(df["Log_Close"].diff().dropna())
print("ADF Statistics:",result[0])
print("P-Values",result[1])
plt.show()

