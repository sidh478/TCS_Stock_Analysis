#importing library
import streamlit as st #for deployment
import pandas as pd #data manipulation and analysis
import numpy as np #mathematical operations
import matplotlib.pyplot as plt 
import seaborn as sns #visualizations
from sklearn.preprocessing import MinMaxScaler #scaling data
import pickle #creating pickle file
from tensorflow.keras.models import Sequential #build neural network model
from tensorflow.keras.layers import LSTM, Dense #build layers for neural network
import warnings
warnings.filterwarnings('ignore') #to ignore warnings in notebook
import nsepy as nse #acess data from National Stock Exchange (NSE)
from datetime import date
import datetime
current_time=datetime.datetime.now() #current time stores present date and time

# load the pre-trained LSTM model 
loaded_model=pickle.load(open(r'C:\Users\harsh\Downloads\LSTM\WEB\trained_model.sav','rb'))

tcs=nse.get_history(symbol='TCS',start=date(2010,1,1),end=date(current_time.year,current_time.month,current_time.day)) #fetching historical past data with start and end date

tcs['Trades']=tcs['Trades'].fillna(method='bfill') ##fill null values with previous values
tcs.drop_duplicates(inplace=True) #droping duplicates

tcs=tcs[['Open','High','Low','Close']] #we are considering columns 


scaler = MinMaxScaler() #calling minmaxscaler
scaled_data = scaler.fit_transform(tcs) #training and transforming scaled data

def tcs_stockprice(n_future):

    n_past = 60 #Past 60 days
    last_30_days = scaled_data[-n_past:]  #uses 60 days as input data (60, 4)
    X_test = np.array([last_30_days]) #reshaping data (1, 60, 4) as model excepts 3D array
    predictions = [] #creating empty list

    for i in range(n_future): # Number of days to forecast
        next_day_pred = loaded_model.predict(X_test)[0, 0] #Predict forecast
        last_30_days = np.append(last_30_days[1:, :], [[next_day_pred, next_day_pred, next_day_pred, next_day_pred]], axis=0) # first 29 values in array are shifted down by one index and predicted value is added to end of array
        X_test = np.array([last_30_days]) #reshaping data (1, 60, 4)
        pred_value = scaler.inverse_transform([[0, 0, 0, next_day_pred]])[0, 3] #setting first 3 values to zero and transforming the last value
        predictions.append(pred_value) #appending to the list
        print("Day {}: {}".format(i+1, pred_value)) #printing day wise

    df=pd.DataFrame(predictions,columns=['Close']) #creating data frame
    # Resetting the index of the DataFrame and renaming the index column as "days"
    df = df.reset_index().rename(columns={'index': 'Days'})
    # Adding 1 to the "days" index column to start from 1
    df['Days'] = df['Days'] + 1
    df = df.set_index(df.columns[0]) #setting indeX column as Days
    st.write(df) #display dataframe

    #line plot
    fig1=plt.figure(figsize=(12,6))
    plt.plot(df)
    plt.title(f'Forecast for next {n_future} days')
    plt.ylabel('Close')
    plt.xlabel('Days')
    st.pyplot(fig1) #display line plot

    return 'Thank You,visit again'

    
def main():
    #giving title
    st.title('Forecasting future data web app')

    #getting input variable from users
    n_future=st.text_input('Number of future data to be forecasted')

    diagnosis=''

    #creating button for prediction
    if st.button('Forecast'):
        diagnosis=tcs_stockprice(int(n_future))
    
    st.success(diagnosis)
if __name__ == '__main__':
    main()