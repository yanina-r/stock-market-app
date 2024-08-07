import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

# Функція для завантаження даних
@st.cache
def load_data(assets, start, end):
    data = yf.download(assets, start=start, end=end)['Adj Close']
    return data

# Параметри
assets = ['AAPL', 'AMZN', 'MRNA', 'TSLA']
start_date = st.date_input('Start Date', pd.to_datetime('2019-01-01'))
end_date = st.date_input('End Date', pd.to_datetime('2024-01-01'))

# Завантаження даних
data = load_data(assets, start_date, end_date)

# Обчислення щоденних доходностей
returns = data.pct_change()

# Розрахунок стандартного відхилення доходностей (щоденна волатильність)
daily_volatility = returns.std()

# Річна волатильність
annual_volatility = daily_volatility * np.sqrt(252)

# Створення Streamlit додатку
st.title('Financial Data Analysis')

# Вибір активів для графіків
selected_assets = st.multiselect('Select Assets for Comparison:', assets, default=assets)

# Візуалізація щоденних доходностей для вибраних активів
st.subheader('Daily Returns of Selected Assets')
fig, ax = plt.subplots(figsize=(14, 7))
for asset in selected_assets:
    ax.plot(data.index, returns[asset], label=asset)
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
