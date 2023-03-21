import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
from pickle import load
from pickle import dump
import warnings
warnings.filterwarnings('ignore')
import nsepy as nse
from datetime import date
import datetime
current_time=datetime.datetime.now()

# load the pre-trained LSTM model
loaded_model=pickle.load(open(r'C:\Users\harsh\Downloads\LSTM\WEB\trained_model.sav','rb'))

tcs=nse.get_history(symbol='TCS',start=date(2010,1,1),end=date(current_time.year,current_time.month,current_time.day))

tcs=tcs[['Open','High','Low','Close']]

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(tcs)

def tcs_stockprice(n_future):

    #n_future = input()  # Number of days to forecast
    n_past = 60
    last_30_days = scaled_data[-n_past:]
    X_test = np.array([last_30_days])
    predictions = []

    for i in range(n_future):
        next_day_pred = loaded_model.predict(X_test)[0, 0]
        last_30_days = np.append(last_30_days[1:, :], [[next_day_pred, next_day_pred, next_day_pred, next_day_pred]], axis=0)
        X_test = np.array([last_30_days])
        pred_value = scaler.inverse_transform([[0, 0, 0, next_day_pred]])[0, 3]
        predictions.append(pred_value)
        print("Day {}: {}".format(i+1, pred_value))
    return np.round(predictions,0)


    
def main():
    #giving title
    st.title('Forcasting future data web app')

    #getting input variable from users
    n_future=st.text_input('Number of future data')

    diagnosis=''

    #creating button for prediction
    if st.button('Future days data predicted'):
        diagnosis=tcs_stockprice(int(n_future))
    
    st.success(diagnosis)
if __name__ == '__main__':
    main()
