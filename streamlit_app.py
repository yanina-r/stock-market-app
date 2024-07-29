import streamlit as st
import pandas as pd
import yfinance as yf

st.set_page_config(
    page_title="Stock Market")

ticker_aapl = yf.Ticker("AAPL")
data_aapl = ticker_aapl.history(period="5y")

plt.figure(figsize=(10, 6))
plt.plot(data_amzn.index, data_amzn['Close'], label='AMZN')
plt.title('Price Amazon (AMZN)')
plt.xlabel('Date')
plt.ylabel('Closing price ($)')
plt.legend()
plt.show()

 
