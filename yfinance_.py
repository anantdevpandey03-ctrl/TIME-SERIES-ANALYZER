import yfinance as yf
import pandas as pd
ticker = "RELIANCE.NS"
data = yf.download(ticker,period="5y")
print(data.head(10))
#data.to_csv("RELIANCE_data.csv")
