import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess data
file_path = 'Cleaned_Crypto_Data.csv'
df = pd.read_csv(file_path)

# Clean data
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
df = df.dropna(subset=['Date'])
df['Bitcoin_Price'] = pd.to_numeric(df['Bitcoin_Price'].str.replace(',', '').str.replace('$', ''), errors='coerce')
df['Ethereum_Price'] = pd.to_numeric(df['Ethereum_Price'].str.replace(',', '').str.replace('$', ''), errors='coerce')
df = df.dropna(subset=['Bitcoin_Price', 'Ethereum_Price'])

# Streamlit app
def main():
    st.title('Cryptocurrency Price and Volatility Analysis')

    st.sidebar.header('Select Date Range')
    start_date = st.sidebar.date_input('Start Date', df['Date'].min().date())
    end_date = st.sidebar.date_input('End Date', df['Date'].max().date())

    # Convert sidebar dates to datetime for comparison
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    if filtered_df.empty:
        st.write("No data available for the selected date range.")
    else:
        st.subheader('Price Changes Analysis')

        # Calculate price changes
        filtered_df['Price_Change_BTC'] = filtered_df['Bitcoin_Price'].diff()
        filtered_df['Price_Change_ETH'] = filtered_df['Ethereum_Price'].diff()

        # Option to compare price changes on the same graph
        compare_prices = st.checkbox('Compare Bitcoin and Ethereum Price Changes on the Same Graph')

        if compare_prices:
            st.write('Bitcoin and Ethereum Price Changes')
            st.line_chart(filtered_df.set_index('Date')[['Price_Change_BTC', 'Price_Change_ETH']])
        else:
            # Bitcoin Price Changes
            st.write('Bitcoin Price Changes')
            st.line_chart(filtered_df[['Date', 'Price_Change_BTC']].set_index('Date'))

            # Ethereum Price Changes
            st.write('Ethereum Price Changes')
            st.line_chart(filtered_df[['Date', 'Price_Change_ETH']].set_index('Date'))

        # Display textual values for price changes
        st.write('Textual Summary of Price Changes')
        btc_summary = filtered_df[['Date', 'Price_Change_BTC']].dropna()
        eth_summary = filtered_df[['Date', 'Price_Change_ETH']].dropna()
        st.write('Bitcoin Price Change Summary:')
        st.write(btc_summary.describe())
        st.write('Ethereum Price Change Summary:')
        st.write(eth_summary.describe())

        st.subheader('Volatility Analysis')

        # Calculate volatility
        def calculate_volatility(df, price_col):
            df['Returns'] = df[price_col].pct_change()
            df['Volatility'] = df['Returns'].rolling(window=21).std()  # 21-day rolling window
            return df[['Date', 'Volatility']].dropna()

        df_btc = calculate_volatility(filtered_df, 'Bitcoin_Price')
        df_eth = calculate_volatility(filtered_df, 'Ethereum_Price')

        # Merge volatility data for comparison
        volatility_df = pd.merge(df_btc, df_eth, on='Date', suffixes=('_BTC', '_ETH'))

        # Option to compare volatility on the same graph
        compare_volatility = st.checkbox('Compare Bitcoin and Ethereum Volatility on the Same Graph')

        if compare_volatility:
            st.write('Bitcoin and Ethereum Volatility')
            st.line_chart(volatility_df.set_index('Date')[['Volatility_BTC', 'Volatility_ETH']])
        else:
            # Bitcoin Volatility
            st.write('Bitcoin Volatility')
            st.line_chart(df_btc.set_index('Date')['Volatility'])

            # Ethereum Volatility
            st.write('Ethereum Volatility')
            st.line_chart(df_eth.set_index('Date')['Volatility'])

        # Display textual values for volatility
        st.write('Textual Summary of Volatility')
        st.write('Bitcoin Volatility Summary:')
        st.write(df_btc[['Date', 'Volatility']].dropna().describe())
        st.write('Ethereum Volatility Summary:')
        st.write(df_eth[['Date', 'Volatility']].dropna().describe())

if __name__ == "__main__":
    main()
