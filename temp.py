# This is a web app to predict stock prices of the S&P 500
# For the data, I used the API yfinance
# In this example I used the streamlit framework to show the results
# 
# Conclusion: The best Machine Learning algorithm is: Linear regression and SVM


import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go



st.set_page_config(
    page_title="Market Profile Chart (US S&P 500)",
    layout="wide")

st.title('Stock Analysis Dashboard')
st.write("""In this web application, you can choose the stock you want to forecast the price, based on machine learning algorithms and more!""")
st.header('Current Stock Price')
st.write("""Choose from the sidebar the stock you want to analyse, the days to show, and the interval""")

tickers = ('AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSLA', 'BRK-A', 'NVDA','JPM') 

# Sidebar of Web App

ticker = st.sidebar.selectbox(
    'Choose a S&P 500 Stock',
     tickers)

intervals = st.sidebar.selectbox(
            "Choose the interval",
        ("1m", "5m", "15m", "30m", "1d")
    )


Days = st.sidebar.number_input("How many days to charge? max=90", min_value = 1, max_value = 90 )

stock = yf.Ticker(ticker)
history_data = stock.history(interval = intervals, period = str(Days) + "d")

prices = history_data['Close']
volumes = history_data['Volume']

dateStr = history_data.index.strftime("%d-%m-%Y %H:%M:%S")

fig = go.Figure(data=[
     go.Candlestick(x=dateStr,
                open=history_data['Open'],
                high=history_data['High'],
                low=history_data['Low'],
                close=history_data['Close'])
    ],
 layout=go.Layout(height=650, width=800)      
    )


st.plotly_chart(fig)

st.header('Volume')
st.line_chart(history_data['Volume'])

#FORCASTING 

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

starting = "2015-01-01"
ending = "2022-03-01"
df = yf.download(ticker, start=starting, end=ending)

# print(df.head())

# x = df[['High', 'Low', 'Open', 'Volume']].values
# y = df['Adj Close'].values

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


df_train, df_test = train_test_split(df, test_size=0.25, random_state=33)

split_at = len(df)//4
df_train = df.head(3*split_at)
df_test = df.tail(split_at)

x_train = df_train[['Open', 'Volume']].values
y_train = df_train['Adj Close'].values

x_test = df_test[['Open', 'Volume']].values
y_test = df_test['Adj Close'].values

lr = LinearRegression()
lr.fit(x_train, y_train)
prediction = lr.predict(x_test)

df1 = df.index

st.header('Forecasting using Linear Reg')
st.write('''Here is the performance of linear regression model in orange''')

history_data1 = stock.history(start=starting, end=ending)
dateStr1 = history_data1.index 


real_df = df.copy()
real_df['Subset'] = 'real'


pred_df = df_test.copy()
pred_df['Adj Close'] = prediction
pred_df['Subset'] = 'pred'

df_copy = pd.concat([real_df, pred_df]).reset_index()

fig1 = plt.figure(figsize=(14, 9))
ax = sns.lineplot(data=df_copy, x="Date", y='Adj Close', hue="Subset")

st.pyplot(fig1)

# RANDOM FOREST

st.header('Forecasting using Random Forest')

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=120)
regressor.fit(x_train, y_train)
prediction_randomf = regressor.predict(x_test)

real_df1 = df.copy()
real_df1['Subset'] = 'real'


pred_df1 = df_test.copy()
pred_df1['Adj Close'] = prediction_randomf
pred_df1['Subset'] = 'pred'

df_copy1 = pd.concat([real_df1, pred_df1]).reset_index()

fig2 = plt.figure(figsize=(14, 9))
ax = sns.lineplot(data=df_copy1, x="Date", y='Adj Close', hue="Subset")

st.pyplot(fig2)
#st.line_chart(prediction)

# SVM

st.header('SVM')

from sklearn.svm import SVR

regr = SVR(kernel='linear', C=1000.0)
regr.fit(x_train, y_train)
svm_pred = regr.predict(x_test)

real_df2 = df.copy()
real_df2['Subset'] = 'real'


pred_df2 = df_test.copy()
pred_df2['Adj Close'] = svm_pred
pred_df2['Subset'] = 'pred'

df_copy2 = pd.concat([real_df2, pred_df2]).reset_index()

fig3 = plt.figure(figsize=(14, 9))
ax = sns.lineplot(data=df_copy2, x="Date", y='Adj Close', hue="Subset")
st.pyplot(fig3)




