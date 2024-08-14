import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

# Load and preprocess cryptocurrency data
def load_crypto_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
    df = df.dropna(subset=['Date'])
    df['Bitcoin_Price'] = pd.to_numeric(df['Bitcoin_Price'].str.replace(',', '').str.replace('$', ''), errors='coerce')
    df['Ethereum_Price'] = pd.to_numeric(df['Ethereum_Price'].str.replace(',', '').str.replace('$', ''), errors='coerce')
    df = df.dropna(subset=['Bitcoin_Price', 'Ethereum_Price'])
    return df

# Calculate volatility
def calculate_volatility(df, price_col):
    df['Returns'] = df[price_col].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=21).std()
    return df[['Date', 'Volatility']].dropna()

# Load stock data from Yahoo Finance
def load_stock_data(assets, start_date, end_date):
    data = yf.download(assets, start=start_date, end=end_date)['Adj Close']
    returns = data.pct_change()
    daily_volatility = returns.std()
    annual_volatility = daily_volatility * np.sqrt(252)
    return returns, annual_volatility

def main():
    st.title('Financial Data Analysis')

    # Cryptocurrency Analysis
    st.header('Cryptocurrency Price and Volatility Analysis')

    file_path = 'Cleaned_Crypto_Data.csv'
    df = load_crypto_data(file_path)

    st.sidebar.header('Select Date Range for Cryptocurrency')
    start_date_crypto = st.sidebar.date_input('Start Date', df['Date'].min().date())
    end_date_crypto = st.sidebar.date_input('End Date', df['Date'].max().date())

    start_date_crypto = pd.to_datetime(start_date_crypto)
    end_date_crypto = pd.to_datetime(end_date_crypto)

    filtered_df = df[(df['Date'] >= start_date_crypto) & (df['Date'] <= end_date_crypto)]

    if filtered_df.empty:
        st.write("No data available for the selected date range.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader('Price Changes Analysis')

            filtered_df['Price_Change_BTC'] = filtered_df['Bitcoin_Price'].diff()
            filtered_df['Price_Change_ETH'] = filtered_df['Ethereum_Price'].diff()

            assets_to_compare = st.multiselect('Select Assets to Compare', ['Bitcoin', 'Ethereum'], default=['Bitcoin'])

            if 'Bitcoin' in assets_to_compare and 'Ethereum' in assets_to_compare:
                st.write('Bitcoin and Ethereum Price Changes')
                st.line_chart(filtered_df.set_index('Date')[['Price_Change_BTC', 'Price_Change_ETH']])
            elif 'Bitcoin' in assets_to_compare:
                st.write('Bitcoin Price Changes')
                st.line_chart(filtered_df[['Date', 'Price_Change_BTC']].set_index('Date'))
            elif 'Ethereum' in assets_to_compare:
                st.write('Ethereum Price Changes')
                st.line_chart(filtered_df[['Date', 'Price_Change_ETH']].set_index('Date'))

        with col2:
            st.subheader('Volatility Analysis')

            df_btc = calculate_volatility(filtered_df, 'Bitcoin_Price')
            df_eth = calculate_volatility(filtered_df, 'Ethereum_Price')

            volatility_df = pd.merge(df_btc, df_eth, on='Date', suffixes=('_BTC', '_ETH'))

            compare_volatility = st.checkbox('Compare Bitcoin and Ethereum Volatility on the Same Graph')

            if compare_volatility:
                st.write('Bitcoin and Ethereum Volatility')
                st.line_chart(volatility_df.set_index('Date')[['Volatility_BTC', 'Volatility_ETH']])
            else:
                st.write('Bitcoin Volatility')
                st.line_chart(df_btc.set_index('Date')['Volatility'])

                st.write('Ethereum Volatility')
                st.line_chart(df_eth.set_index('Date')['Volatility'])

    # Stock Data Analysis
    st.header('Stock Returns and Volatility Analysis')

    st.sidebar.header('Select Date Range for Stock')
    start_date_stock = st.sidebar.date_input('Start Date for Stocks', datetime(2019, 1, 1).date())
    end_date_stock = st.sidebar.date_input('End Date for Stocks', datetime(2024, 1, 1).date())

    start_date_stock = pd.to_datetime(start_date_stock)
    end_date_stock = pd.to_datetime(end_date_stock)

    assets = ['AAPL', 'AMZN', 'MRNA', 'TSLA']
    returns, annual_volatility = load_stock_data(assets, start_date_stock, end_date_stock)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Daily Returns Analysis')
        selected_stocks = st.multiselect('Select Stocks to Display', returns.columns.tolist(), default=returns.columns.tolist())
        st.line_chart(returns[selected_stocks])

    with col2:
        st.subheader('Annual Volatility Analysis')
        st.bar_chart(annual_volatility[selected_stocks])

        # Calculate and display stock volatility
        st.subheader('Stock Volatility Analysis')
        stock_volatility_df = pd.DataFrame({
            'Date': returns.index,
            'Volatility': returns[selected_stocks].rolling(window=21).std().mean(axis=1)
        }).dropna()

        st.line_chart(stock_volatility_df.set_index('Date')['Volatility'])

if __name__ == "__main__":
    main()
