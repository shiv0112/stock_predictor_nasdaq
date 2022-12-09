import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pandas.tseries.offsets import DateOffset

st.title('NASDAQ Stock Prediction')
st.text('This is a web app to allow users to forecast value of NASDAQ stocks')

df=pd.read_csv('data.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df.set_index('DATE',inplace=True)
df=df.resample('B').mean().dropna()

s_stock=0
s_stock = st.selectbox(
    'Which stock would you like to forecast?',
    ('Select a stock','NASDAQ.AAPL','NASDAQ.ADP','NASDAQ.CBOE','NASDAQ.CSCO','NASDAQ.EBAY'))

if s_stock!='Select a stock':
    st.header(s_stock+' Dataset')
    st.dataframe(df[s_stock].head())
    df[s_stock].plot(figsize=(12,4))
    plt.ylabel('Stock Value')
    plt.title(s_stock)
    st.pyplot(plt)
    st.write(s_stock)
    df= df[[s_stock]]

    list_model=['aapl.pkl','adp.pkl','cboe.pkl','csco.pkl','ebay.pkl']
    list_stock=['NASDAQ.AAPL','NASDAQ.ADP','NASDAQ.CBOE','NASDAQ.CSCO','NASDAQ.EBAY']
    ind=list_stock.index(str(s_stock))
    
    model = sm.load(list_model[ind])
    
    st.header('Input Slider')
    step = st.slider("Select Months to Forecast:",1,12)
    
    st.header('Forecast')
    if step!=1:
        future_dates=[df.index[-1]+ DateOffset(months=x)for x in range(0,step+1)]
        future_data=pd.DataFrame(index=future_dates[1:],columns=[s_stock])
        forecast=model.forecast(steps=step)
        future_stocks = pd.DataFrame(forecast).set_index(future_data.index)
        future_stocks.rename(columns = {'predicted_mean':s_stock}, inplace = True)

        pd.concat([df,future_stocks]).plot(figsize=(12,4))
        plt.ylabel('Sales')
        plt.title("Predicted")
        plt.xlabel('Time')
        plt.axvline('2017-08-31',color='orange',lw=2,ls='dashed')
        plt.legend(loc=0)
        st.pyplot(plt)

