import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

# Функція для завантаження даних з Yahoo Finance
@st.cache
def load_stock_data(assets, start, end):
    data = yf.download(assets, start=start, end=end)['Adj Close']
    return data

# Функція для завантаження даних про криптовалюти з CSV

@st.cache
def load_crypto_data('Stock Market Dataset.csv'):
    if os.path.exists('Stock Market Dataset.csv'):
        crypto_data = pd.read_csv('Stock Market Dataset.csv', parse_dates=['Date'], index_col='Date')
        return crypto_data
    else:
        st.error(f"File not found: {'Stock Market Dataset.csv'}")
        return pd.DataFrame()

# Параметри
stock_assets = ['AAPL', 'AMZN', 'MRNA', 'TSLA']
start_date = st.date_input('Start Date', pd.to_datetime('2019-01-01'))
end_date = st.date_input('End Date', pd.to_datetime('2024-01-01'))
crypto_file_path = st.text_input('Stock Market Dataset.csv')

# Завантаження даних
stock_data = load_stock_data(stock_assets, start_date, end_date)
crypto_data = load_crypto_data(crypto_file_path)

# Об'єднання даних акцій і криптовалют
data = pd.concat([stock_data, crypto_data], axis=1).dropna()

# Обчислення щоденних доходностей
returns = data.pct_change()

# Розрахунок стандартного відхилення доходностей (щоденна волатильність)
daily_volatility = returns.std()

# Річна волатильність
annual_volatility = daily_volatility * np.sqrt(252)

# Створення Streamlit додатку
st.title('Financial Data Analysis')

# Вибір активів для графіків
all_assets = stock_assets + ['BTC', 'ETH']
selected_assets = st.multiselect('Select Assets for Comparison:', all_assets, default=all_assets)

# Вибір типу графіку
chart_type = st.selectbox('Select Chart Type:', ['Line Chart', 'Bar Chart'])

# Фільтр по періодам для щоденних доходностей
selected_period = st.selectbox('Select Period for Daily Returns:', ['All', 'Last 6 Months', 'Last 1 Year'])

if selected_period == 'Last 6 Months':
    filtered_data = data.loc[data.index >= (pd.to_datetime(end_date) - pd.DateOffset(months=6))]
elif selected_period == 'Last 1 Year':
    filtered_data = data.loc[data.index >= (pd.to_datetime(end_date) - pd.DateOffset(years=1))]
else:
    filtered_data = data

filtered_returns = filtered_data.pct_change()

# Візуалізація щоденних доходностей для вибраних активів
st.subheader('Daily Returns of Selected Assets')
fig, ax = plt.subplots(figsize=(14, 7))
for asset in selected_assets:
    if chart_type == 'Line Chart':
        ax.plot(filtered_data.index, filtered_returns[asset], label=asset)
    elif chart_type == 'Bar Chart':
        ax.bar(filtered_data.index, filtered_returns[asset], label=asset)
ax.set_xlabel('Date')
ax.set_ylabel('Daily Returns')
ax.set_title('Daily Returns of Selected Assets')
ax.legend()
st.pyplot(fig)

# Візуалізація річної волатильності для вибраних активів
st.subheader('Annual Volatility of Selected Assets')
fig, ax = plt.subplots(figsize=(10, 6))
annual_volatility[selected_assets].plot(kind='bar', ax=ax, title='Annual Volatility of Selected Assets')
ax.set_ylabel('Annual Volatility')
st.pyplot(fig)

# Показувати деталі річної волатильності для обраних активів
st.subheader('Annual Volatility Details')
volatility_details = pd.DataFrame({
    'Asset': annual_volatility.index,
    'Annual Volatility': annual_volatility.values
})
st.dataframe(volatility_details)

# Порівняння волатильності акцій і криптовалют
st.subheader('Comparison of Stock and Cryptocurrency Volatility')
stocks_volatility = annual_volatility[stock_assets].mean()
crypto_volatility = annual_volatility[['BTC', 'ETH']].mean()

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(['Stocks', 'Cryptocurrencies'], [stocks_volatility, crypto_volatility], color=['blue', 'orange'])
ax.set_ylabel('Annual Volatility')
ax.set_title('Comparison of Stock and Cryptocurrency Volatility')
st.pyplot(fig)

# Визначення ризиковості активів
st.subheader('Risk Analysis')
risk_threshold = st.slider('Select Volatility Threshold:', 0.0, 1.0, 0.3)
risky_assets = annual_volatility[annual_volatility > risk_threshold].index.tolist()
less_risky_assets = annual_volatility[annual_volatility <= risk_threshold].index.tolist()

st.write(f"Assets more risky than {risk_threshold}: {', '.join(risky_assets)}")
st.write(f"Assets less risky than {risk_threshold}: {', '.join(less_risky_assets)}")
