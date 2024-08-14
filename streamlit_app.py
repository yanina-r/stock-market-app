import streamlit as st
import pandas as pd
import numpy as np
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
    daily_volatility = returns.rolling(window=21).std()
    annual_volatility = daily_volatility * np.sqrt(252)
    return data, daily_volatility, annual_volatility

def main():
    st.title('Market Volatility Analysis')

    tabs = st.tabs(["Cryptocurrency Analysis", "Stock Market Analysis", "Comparison of Cryptocurrency and Stock Volatility"])

    with tabs[0]:
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
            st.subheader('Price Changes Analysis')

            filtered_df['Price_Change_BTC'] = filtered_df['Bitcoin_Price'].diff()
            filtered_df['Price_Change_ETH'] = filtered_df['Ethereum_Price'].diff()

            assets_to_compare = st.multiselect('Select Assets to Compare', ['Bitcoin', 'Ethereum'], default=['Bitcoin', 'Ethereum'])

            col1, col2 = st.columns(2)
            with col1:
                if 'Bitcoin' in assets_to_compare:
                    st.write('Bitcoin Price Changes')
                    st.line_chart(filtered_df[['Date', 'Price_Change_BTC']].set_index('Date'))
            with col2:
                if 'Ethereum' in assets_to_compare:
                    st.write('Ethereum Price Changes')
                    st.line_chart(filtered_df[['Date', 'Price_Change_ETH']].set_index('Date'))

            st.subheader('Volatility Analysis')
            df_btc = calculate_volatility(filtered_df, 'Bitcoin_Price')
            df_eth = calculate_volatility(filtered_df, 'Ethereum_Price')

            col3, col4 = st.columns(2)
            with col3:
                if 'Bitcoin' in assets_to_compare:
                    st.write('Bitcoin Volatility')
                    st.line_chart(df_btc.set_index('Date')['Volatility'])
            with col4:
                if 'Ethereum' in assets_to_compare:
                    st.write('Ethereum Volatility')
                    st.line_chart(df_eth.set_index('Date')['Volatility'])

    with tabs[1]:
        st.header('Stock Returns and Volatility Analysis')

        st.sidebar.header('Select Date Range for Stock')
        start_date_stock = st.sidebar.date_input('Start Date for Stocks', datetime(2019, 1, 1).date())
        end_date_stock = st.sidebar.date_input('End Date for Stocks', datetime(2024, 1, 1).date())

        start_date_stock = pd.to_datetime(start_date_stock)
        end_date_stock = pd.to_datetime(end_date_stock)

        assets = ['AAPL', 'AMZN', 'MRNA', 'TSLA']
        stock_data, daily_volatility, annual_volatility = load_stock_data(assets, start_date_stock, end_date_stock)

        selected_stocks = st.multiselect('Select Stocks to Display', stock_data.columns.tolist(), default=stock_data.columns.tolist())

        col5, col6 = st.columns(2)
        with col5:
            st.write('Stock Price Changes')
            st.line_chart(stock_data[selected_stocks])
        with col6:
            st.write('Stock Volatility')
            st.line_chart(daily_volatility[selected_stocks])

    with tabs[2]:
        st.header('Comparison of Cryptocurrency and Stock Volatility')

        selected_crypto = st.multiselect('Select Cryptocurrencies to Compare', ['Bitcoin', 'Ethereum'], default=['Bitcoin', 'Ethereum'])
        selected_stocks = st.multiselect('Select Stocks to Compare', assets, default=assets)

        if selected_crypto and selected_stocks:
            # Calculate volatility for selected cryptocurrencies
            crypto_vol = pd.DataFrame()
            for name in selected_crypto:
                vol_df = calculate_volatility(df, f'{name}_Price')
                crypto_vol[name] = vol_df.set_index('Date')['Volatility']

            # Combine stock and crypto volatility data
            combined_vol_df = pd.concat([crypto_vol, daily_volatility[selected_stocks]], axis=1)

            # Plot comparison chart
            st.write('Comparison of Cryptocurrency and Stock Volatility')
            st.line_chart(combined_vol_df)

    # Increase spacing between columns by adding margin to columns
    st.markdown("""
        <style>
            .css-1n07l7f {margin-left: 20px; margin-right: 20px;}
        </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
