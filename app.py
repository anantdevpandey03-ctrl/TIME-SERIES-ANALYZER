import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go

st.title("📈 Stock Price Forecasting Dashboard")
st.write("ARIMA based 30-day stock price forecast")

# Sidebar
ticker = st.sidebar.text_input("Enter Stock Ticker", value="RELIANCE.NS")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2021-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2026-01-01"))

if ticker:
    # Data fetch
    st.subheader("📊 Historical Price")
    data = yf.download(ticker, start=start_date, end=end_date)
    data.columns = data.columns.droplevel(1)  # ← add karo
    series = data["Close"].dropna()
    series.index = pd.to_datetime(series.index)  # ← add karo
    # Historical graph
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=series.index, y=series.values, name="Close Price"))
    fig1.update_layout(xaxis_title="Date", yaxis_title="Price (INR)")
    st.plotly_chart(fig1)

    # Train/Test split
    train_size = int(len(series) * 0.8)
    train = series[:train_size]
    test = series[train_size:]

    # Model
    model = ARIMA(train, order=(0,1,0))
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=len(test))

    # Metrics
    mae = mean_absolute_error(test, predictions)
    rmse = np.sqrt(mean_squared_error(test, predictions))

    col1, col2 = st.columns(2)
    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("RMSE", f"{rmse:.2f}")

    # Actual vs Predicted
    st.subheader("🔍 Actual vs Predicted")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=train.index, y=train.values, name="Train"))
    fig2.add_trace(go.Scatter(x=test.index, y=test.values, name="Actual"))
    fig2.add_trace(go.Scatter(x=test.index, y=predictions.values, name="Predicted"))
    st.plotly_chart(fig2)

    # 30 Day Forecast
    st.subheader("🚀 30-Day Future Forecast")
    model2 = ARIMA(series, order=(0,1,0))
    model_fit2 = model2.fit()
    forecast = model_fit2.forecast(steps=30)
    future_dates = pd.bdate_range(start=series.index[-1] + pd.Timedelta(days=1), periods=30)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=series.index, y=series.values, name="Actual Price"))
    fig3.add_trace(go.Scatter(x=future_dates, y=forecast.values, name="30-Day Forecast"))
    st.plotly_chart(fig3)
#streamlit run app.py