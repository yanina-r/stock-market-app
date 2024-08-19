import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

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

#Forecasting
# Function to load and prepare cryptocurrency data
def load_and_prepare_crypto_data():
    file_path = 'Cleaned_Crypto_Data.csv'
    df = pd.read_csv(file_path)

    # Data cleaning
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
    df = df.dropna(subset=['Date'])
    df['Bitcoin_Price'] = pd.to_numeric(df['Bitcoin_Price'].str.replace(',', '').str.replace('$', ''), errors='coerce')
    df['Ethereum_Price'] = pd.to_numeric(df['Ethereum_Price'].str.replace(',', '').str.replace('$', ''), errors='coerce')
    df = df.dropna(subset=['Bitcoin_Price', 'Ethereum_Price'])

    # Feature engineering
    df['Bitcoin_Price_Change'] = df['Bitcoin_Price'].diff()
    df['Ethereum_Price_Change'] = df['Ethereum_Price'].diff()
    df = df.dropna(subset=['Bitcoin_Price_Change', 'Ethereum_Price_Change'])

    # Prepare features and target variables
    X = df[['Bitcoin_Price', 'Ethereum_Price']]
    y_btc = df['Bitcoin_Price_Change']
    y_eth = df['Ethereum_Price_Change']
    dates = df['Date']

    X_train_btc, X_test_btc, y_train_btc, y_test_btc, dates_train_btc, dates_test_btc = train_test_split(X, y_btc, dates, test_size=0.3, random_state=42)
    X_train_eth, X_test_eth, y_train_eth, y_test_eth, dates_train_eth, dates_test_eth = train_test_split(X, y_eth, dates, test_size=0.3, random_state=42)

    return X_train_btc, X_test_btc, y_train_btc, y_test_btc, dates_train_btc, dates_test_btc, X_train_eth, X_test_eth, y_train_eth, y_test_eth, dates_train_eth, dates_test_eth

# Function to load and prepare stock data
def load_and_prepare_stock_data(assets):
    data = yf.download(assets, start='2019-01-01', end='2024-01-01')['Adj Close']
    price_changes = data.pct_change().dropna()

    # Prepare features and target variables
    X = price_changes.shift(1).dropna()
    y = price_changes.iloc[1:]

    if X.empty or y.empty:
        st.write("Data for training and testing is empty. Please check the data preparation steps.")
        return None, None, None, None

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    except ValueError as e:
        st.write(f"Error: {e}")
        st.write("Adjusting the test size...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=False)

    dates_train = X_train.index
    dates_test = X_test.index

    return X_train, X_test, y_train, y_test, dates_train, dates_test

# Function to forecast cryptocurrencies
def forecast_crypto(selected_assets):
    X_train_btc, X_test_btc, y_train_btc, y_test_btc, dates_train_btc, dates_test_btc, X_train_eth, X_test_eth, y_train_eth, y_test_eth, dates_train_eth, dates_test_eth = load_and_prepare_crypto_data()

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    results = {}

    for model_name, model in models.items():
        if 'Bitcoin' in selected_assets:
            model.fit(X_train_btc, y_train_btc)
            y_pred_btc = model.predict(X_test_btc)
            mse_btc = mean_squared_error(y_test_btc, y_pred_btc)
            r2_btc = r2_score(y_test_btc, y_pred_btc)
            
            results[f'Bitcoin_{model_name}_MSE'] = mse_btc
            results[f'Bitcoin_{model_name}_R2'] = r2_btc

            # Plot predictions for Bitcoin
            st.subheader(f'Bitcoin - Actual vs Predicted ({model_name})')
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.scatter(dates_test_btc, y_test_btc, color='blue', label='Actual Price Change')
            ax.scatter(dates_test_btc, y_pred_btc, color='red', label='Predicted Price Change')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price Change')
            ax.set_title(f'Bitcoin - Actual vs Predicted ({model_name})')
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45)
            st.pyplot(fig)

        if 'Ethereum' in selected_assets:
            model.fit(X_train_eth, y_train_eth)
            y_pred_eth = model.predict(X_test_eth)
            mse_eth = mean_squared_error(y_test_eth, y_pred_eth)
            r2_eth = r2_score(y_test_eth, y_pred_eth)
            
            results[f'Ethereum_{model_name}_MSE'] = mse_eth
            results[f'Ethereum_{model_name}_R2'] = r2_eth

            # Plot predictions for Ethereum
            st.subheader(f'Ethereum - Actual vs Predicted ({model_name})')
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.scatter(dates_test_eth, y_test_eth, color='blue', label='Actual Price Change')
            ax.scatter(dates_test_eth, y_pred_eth, color='red', label='Predicted Price Change')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price Change')
            ax.set_title(f'Ethereum - Actual vs Predicted ({model_name})')
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45)
            st.pyplot(fig)

    # Display results
    for key, value in results.items():
        st.write(f'{key}: {value}')

# Function to forecast stocks
def forecast_stocks(selected_assets):
    assets = [asset for asset in selected_assets if asset in ['AAPL', 'AMZN', 'MRNA', 'TSLA']]
    if not assets:
        st.write("No valid stock assets selected.")
        return

    X_train, X_test, y_train, y_test, dates_train, dates_test = load_and_prepare_stock_data(assets)

    if X_train is None:
        return

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    results = {}

    for model_name, model in models.items():
        for asset in assets:
            model.fit(X_train, y_train[asset])
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test[asset], y_pred)
            r2 = r2_score(y_test[asset], y_pred)
            
            results[f'{asset}_{model_name}_MSE'] = mse
            results[f'{asset}_{model_name}_R2'] = r2

            # Plot predictions for stocks
            st.subheader(f'{asset} - Actual vs Predicted ({model_name})')
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(dates_test, y_test[asset], label='Actual Price Change', color='blue')
            ax.plot(dates_test, y_pred, label='Predicted Price Change', color='red')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price Change')
            ax.set_title(f'{asset} - Actual vs Predicted ({model_name})')
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45)
            st.pyplot(fig)

    # Display results
    for key, value in results.items():
        st.write(f'{key}: {value}')



def main():
    st.title('Market Volatility Analysis')

    tabs = st.tabs(["Cryptocurrency Analysis", "Stock Market Analysis", "Comparison of Cryptocurrency and Stock Volatility", "Data Forecasting"])

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

    with tabs[3]:
        st.header('Data Forecasting')

        st.sidebar.header('Select Assets for Forecasting')
        all_assets = ['Bitcoin', 'Ethereum', 'AAPL', 'AMZN', 'MRNA', 'TSLA']
        selected_assets = st.sidebar.multiselect('Select Assets for Forecasting', all_assets, default=['Bitcoin'])

        if selected_assets:
            st.header('Forecasting Results') 
        if any(asset in selected_assets for asset in ['Bitcoin', 'Ethereum']):
            forecast_crypto(selected_assets)
        if any(asset in selected_assets for asset in ['AAPL', 'AMZN', 'MRNA', 'TSLA']):
            forecast_stocks(selected_assets)
        else:
            st.write("Please select at least one asset for forecasting.")


if __name__ == "__main__":
    main()
