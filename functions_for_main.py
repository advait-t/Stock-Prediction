import os
import time
from dateutil.parser import parse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

#! Function to check if there is any new company in the list or an old company has been removed from the list
def check_for_changes_in_companies(training_data_path, companies_list_path):
    existing_company_list = pd.read_csv(training_data_path)["Company"].unique()
    with open(companies_list_path, "r") as f:
        new_companies_list=[i for line in f for i in line.split(',')]

    new_company = list(set(new_companies_list) - set(existing_company_list))
    delete_company = list(set(existing_company_list) - set(new_companies_list))
    return new_company, delete_company

#! Function to fetch data for new company from yahoo finance
def YahooFinanceHistory(company, previous_days, training_data_path):
    '''
    
    This function takes the company name and the number of previous days as input and returns the dataframe of the company history.

    Variables:

    company: string, name of the company
    previous_days: int, number of days to extract data from
    today: date, today's date
    past: date, date of the past
    query_string: string, query string to extract data from yahoo finance
    company_prices: dataframe, dataframe containing the prices of the company
    company_data: dataframe, dataframe containing the data of the company
    valuation_measures: list, list containing the valuation measures interested in
    company_valuation: dataframe, dataframe containing the valuation measures of the company
    path_save_as_csv: boolean, True if the dataframe is to be saved as a csv file, False otherwise
    
    '''
    
    # today = int(time.mktime((datetime.now()).timetuple()))
    # past = int(time.mktime((datetime.now() - timedelta(previous_days)).timetuple()))
    
    # interval = '1d'

    # # defining the query to get historical stock data
    # query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{company}?period1={past}&period2={today}&interval={interval}&events=history&includeAdjustedClose=true'
    
    # company_prices = pd.read_csv(query_string)


    today = date.today()
    past = today - timedelta(previous_days)
    company_prices = yf.download(company, start = past, end = today)
    company_prices = company_prices.reset_index()
    # company_prices = company_prices[['Date', 'Close']]
    # company_prices.columns = ['Date', 'Close']
    company_prices['Date'] = pd.to_datetime(company_prices['Date'])
    company_prices = company_prices.sort_values(by = 'Date')
    company_prices = company_prices.reset_index(drop = True)

    company_prices['Company'] = company
    training_data = pd.read_csv(training_data_path)

    training_data = training_data.append(company_prices)

    training_data1 = training_data[training_data['Company'] == company]
    training_data = training_data[training_data['Company'] != company]

    if training_data1['Date'].tail(1).values[0] != company_prices['Date'].tail(1).values[0]: 
        training_data1 = training_data1.append(company_prices.tail(1))
    else:
        pass

    training_data = training_data.append(training_data1)
    data = training_data[training_data['Company'] == company]
    data1 = training_data[training_data['Company'] != company]
    data.drop_duplicates(subset = 'Date', inplace = True, keep = 'last')
    data.reset_index(inplace = True, drop = True)
    training_data = data1.append(data)
    training_data.to_csv(training_data_path, index = False)

    return company_prices


#! Function to read data from csv file
def read_data(company, previous_days, training_data_path, holidays_list_path = 0):

    company_prices = YahooFinanceHistory(company, previous_days, training_data_path)
    company_prices = company_prices[:-1]
    company_prices = company_prices[['Date', 'Close']]
    company_prices.columns = ['ds', 'y']
    company_prices['ds'] = pd.to_datetime(company_prices['ds'])

    holidays_list = pd.read_csv(holidays_list_path)

    for i in range(len(holidays_list['Day'])):
        holidays_list['Day'][i] = pd.to_datetime(parse(holidays_list['Day'][i]))

    holidays_list = holidays_list[['Holiday','Day']]
    holidays_list = holidays_list.rename({'Day':'ds', 'Holiday':'holiday'}, axis = 1)   

    return company_prices, holidays_list

def fetch_data_new_company(new_company, training_data_path, holidays_list_path):
    new_company = ','.join(new_company)
    new_company_prices = read_data(new_company, 365*5, training_data_path,holidays_list_path) # read data for 5 years
    
    return new_company_prices

def data_delete_old_company(old_company, training_data_path, error_df_path, model_path):
    # old_company = ','.join(old_company)
    training_data = pd.read_csv(training_data_path)
    training_data = training_data[training_data['Company'] != old_company]
    training_data.to_csv(training_data_path, index=False)
    os.remove(error_df_path + old_company + '.csv')
    os.remove(model_path + old_company + '.json')
    
 

import csv
import json
import time
import prophet
import warnings
import numpy as np
import pandas as pd
import streamlit as st
from dateutil.parser import parse
from datetime import datetime, timedelta, date
from prophet.serialize import model_to_json, model_from_json
from sklearn import metrics
from dateutil.parser import parse
from prophet import Prophet
import warnings
warnings.filterwarnings("ignore")


#! Store the model performance metrics in a file to compare the performance of different models

def model_building_for_new_company(company, company_prices, holidays_list, h, eliminate_weekends, model_path, error_df_path, error_metrics_path):

    if holidays_list is not None:

        # variables for the model building and their meaning:
        '''
        holidays: list, list of holidays
        n_changepoints: int, number of changepoints. Change points are abrupt variations in time series data. (n_changepoints = 1 means there is only one changepoint.)
        n_changepoints_scale: float, scale of the number of changepoints 
        changepoint_prior_scale: float, scale of the changepoint prior
        yearly_seasonality: boolean, True if yearly seasonality is to be used, False otherwise
        weekly_seasonality: boolean, True if weekly seasonality is to be used, False otherwise
        daily_seasonality: boolean, True if daily seasonality is to be used, False otherwise
        holidays_prior_scale: float, scale of the holiday prior
        holidays_yearly_prior_scale: float, scale of the yearly holiday prior
        fourier_order: int, order of the fourier series. How quickly the seasonility of the time series can change.
        '''

        m = Prophet(growth="linear",
            holidays= holidays_list,
            seasonality_mode="multiplicative",
            changepoint_prior_scale=30,
            seasonality_prior_scale=35,
            holidays_prior_scale=20,
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False,
            ).add_seasonality(
                name='monthly',
                period=30.5,
                fourier_order=55
            ).add_seasonality(
                name="daily",
                period=1,
                fourier_order=15
            ).add_seasonality(
                name="weekly",
                period=7,
                fourier_order=20
            ).add_seasonality(
                name="yearly",
                period=365.25,
                fourier_order=20
            ).add_seasonality(
                name="quarterly",
                period = 365.25/4,
                fourier_order=5,
                prior_scale = 15)
    else:
        m = Prophet(growth = 'linear')

    # make last 30 days of the data as the test data and remove them from the training data
    test_data = company_prices[-30:]

    company_prices = company_prices[:-30]

    

    model = m.fit(company_prices)

    future_dates = model.make_future_dataframe(periods = h)

    if eliminate_weekends is not None:
        future_dates['day'] = future_dates['ds'].dt.weekday
        future_dates = future_dates[future_dates['day']<=4]

    
    #! saving the model
    with open(model_path + company + '.json', 'w') as fout:
        json.dump(model_to_json(model), fout)  # Save model

    #! Creating a dataframe for the new company which will log all the values for prediction and track errors
    error_df = pd.DataFrame(columns=['Date', 'Actual_Close', 'Predicted_Close', 'Predicted_Close_Minimum', 'Predicted_Close_Maximum', 'Percent_Change_from_Close', 'Actual_Up_Down', 'Predicted_Up_Down', 'Company'])
    error_df = error_df.append({'Date': '07-04-2022'}, ignore_index=True)
    error_df.to_csv(error_df_path + company + '.csv', index=False)

    # Testing the model on the test set and calculating the error metrics and saving to a csv file
    # The error metrics are calculated for the last 30 days of the test set and comparing between the actual and predicted values
    
    forecast = model.predict(test_data)
    forecast = forecast[['ds', 'yhat']]
    forecast = pd.merge(forecast, test_data, how='left', left_on='ds', right_on='ds')
    forecast = forecast.rename(columns={'ds': 'Date', 'yhat': 'Predicted_Close', 'y': 'Actual_Close'})

    # calculate the RMSE
    rmse = np.sqrt(metrics.mean_squared_error(forecast['Actual_Close'], forecast['Predicted_Close']))
    # calculate the MAPE
    mape = np.mean(np.abs(forecast['Actual_Close'] - forecast['Predicted_Close'])/np.abs(forecast['Actual_Close']))
    # calculate the MAE
    mae = metrics.mean_absolute_error(forecast['Actual_Close'], forecast['Predicted_Close'])
    # calculate the R2
    r2 = metrics.r2_score(forecast['Actual_Close'], forecast['Predicted_Close'])

    error_metrics = pd.DataFrame(columns=['RMSE', 'MAPE', 'MAE', 'R2', 'Company'])
    error_metrics = error_metrics.append({'RMSE': rmse, 'MAPE': mape, 'MAE': mae, 'R2': r2, 'Company': company}, ignore_index=True)
    error_metrics.to_csv(error_metrics_path + 'error_metrics.csv', index=False)

    prediction = model.predict(future_dates)

    return model, prediction, future_dates

  
import os
import warnings
import pandas as pd
from datetime import date
from data_fetching import *
from model_building import *
warnings.filterwarnings("ignore")

#! Loading Model
def load_model(model_path):
    with open(model_path, 'r') as fin:
        saved_model = model_from_json(json.load(fin))  # Load model
    return saved_model

#! check for holiday
def is_holiday(today):
    # holidays_list = pd.read_csv('/Users/advait_t/Desktop/Jio/Stock_Prediction/Stock_Prediction/data/final/2017-2022_Holidays_NSE_BSE_EQ_EQD.csv')
    holidays_list = pd.read_csv('/Users/advait_t/Desktop/Jio/Stock_Prediction/Stock_Prediction/data/final/2017-2022_Holidays_NSE_BSE_EQ_EQD.csv')
    for i in range(len(holidays_list['Day'])):
        holidays_list['Day'][i] = pd.to_datetime(parse(holidays_list['Day'][i]))
    for i in range(len(holidays_list['Day'])):
        if holidays_list['Day'][i].date() == today:
            return True
    return False

#! get real stock price
def real_stock_price(company, predicted):

    now = datetime.now()
    weekday_weekend = datetime.strptime(str(predicted['ds'][0]), '%Y-%m-%d %H:%M:%S')
    
    if weekday_weekend.weekday() <= 5 and weekday_weekend.weekday() != 0:
        days = 1
    elif weekday_weekend.weekday() == 6:
        days = 2
    elif weekday_weekend.weekday() == 0:
        days = 3

    past = datetime.strptime(str(predicted['ds'][0]), '%Y-%m-%d %H:%M:%S') - timedelta(days)
    past = past.replace(hour = now.hour, minute = now.minute, second = now.second, microsecond = now.second)
    past = int(time.mktime(past.timetuple()))
    
    interval = '1d'

    # defining the query to get historical stock data
    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{company}?period1={past}&period2={past}&interval={interval}&events=history&includeAdjustedClose=true'
    
    try:
        company_stock_price = pd.read_csv(query_string)
        company_stock_price = company_stock_price[['Date', 'Close']]
        return company_stock_price
    except:
        company_stock_price = pd.DataFrame(np.nan, index = [0], columns=['Date'])
        return company_stock_price

#! for next day prediction
def next_day_prediction(model_path, missing_dates, missing_dates_df = 0):

    saved_model = load_model(model_path)

    if missing_dates == False:
        next_day = date.today() + timedelta(days=1)
        future_date = pd.DataFrame(pd.date_range(start = next_day, end = next_day, freq ='D'), columns = ['ds'])
        predicted = saved_model.predict(future_date)
        return (predicted[['ds','yhat', 'yhat_upper', 'yhat_lower']])
    else:
        missing_dates_df.rename(columns={'Date':'ds'}, inplace=True)
        predicted = saved_model.predict(missing_dates_df)
        return (predicted[['ds','yhat', 'yhat_upper', 'yhat_lower']])

def real_stock_price_missing_date(company, predicted):
    now = datetime.now()
    predicted['Close'] = None
    for i in range(len(predicted['ds'])):
        past = datetime.strptime(str(predicted['ds'][i]), '%Y-%m-%d %H:%M:%S')
        past = past.replace(hour = now.hour, minute = now.minute, second = now.second, microsecond = now.second)
        past = int(time.mktime(past.timetuple()))
        interval = '1d'
        query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{company}?period1={past}&period2={past}&interval={interval}&events=history&includeAdjustedClose=true'
        company_stock_price = pd.read_csv(query_string)
        company_stock_price = company_stock_price[['Date', 'Close']]
        predicted['Close'][i] = company_stock_price['Close'].values[0]
    return predicted

#! Filling missing dates
def filling_missing_dates(error_df, company):
    Date = date.today()
    
    date_range = pd.date_range(start = error_df.iloc[-1]['Date'], end = Date, freq ='B')

    date_range_df = pd.DataFrame(columns = error_df.columns)
    date_range_df['Date'] = date_range
    date_range_df['Date'] = date_range_df['Date'].dt.date

    for i in range(len(date_range_df['Date'])):
        if is_holiday(date_range_df['Date'][i]) == True:
            date_range_df = date_range_df[date_range_df['Date'] != date_range_df['Date'][i]]
            
    missing_dates_df = next_day_prediction(f'/Users/advait_t/Desktop/Jio/Stock_Prediction/Stock_Prediction/models/{company}.json',True, date_range_df)
    missing_dates_df = real_stock_price_missing_date(company, missing_dates_df)

    # convert ds from datetime to date
    missing_dates_df['ds'] = missing_dates_df['ds'].dt.date

    missing_dates_df.rename(columns = {'ds':'Date', 'Close':'Actual_Close', 'yhat':'Predicted_Close', 'yhat_upper':'Predicted_Close_Maximum', 'yhat_lower':'Predicted_Close_Minimum'}, inplace = True)
    missing_dates_df['Percent_Change_from_Close'] = ((missing_dates_df['Actual_Close'] - missing_dates_df['Predicted_Close'])/missing_dates_df['Actual_Close'])*100

    missing_dates_df['Actual_Up_Down'] = np.where((missing_dates_df['Actual_Close'] > missing_dates_df['Actual_Close'].shift(-1)), 'Up', 'Down')
    missing_dates_df['Predicted_Up_Down'] = np.where((missing_dates_df['Predicted_Close'] > missing_dates_df['Actual_Close'].shift(-1)), 'Up', 'Down')

    error_df = error_df.append(missing_dates_df, ignore_index= True)
    error_df = error_df.drop_duplicates(subset = 'Date', keep = 'last')
    error_df['Company'] = company

    error_df['Actual_Close'] = error_df['Actual_Close'].astype(float)
    error_df['Predicted_Close'] = error_df['Predicted_Close'].astype(float)
    error_df['Predicted_Close_Minimum'] = error_df['Predicted_Close_Minimum'].astype(float)
    error_df['Predicted_Close_Maximum'] = error_df['Predicted_Close_Maximum'].astype(float)
    error_df['Percent_Change_from_Close'] = error_df['Percent_Change_from_Close'].astype(float)
    return error_df

def pred_vs_real_comparision(real_stock_price, predicted, error_df, company):

    df = pd.DataFrame([[np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]], columns=error_df.columns)
    error_df = pd.concat([error_df, df], ignore_index =True)

    error_df['Date'].iloc[-1] = str(predicted['ds'].iloc[-1].strftime('%Y-%m-%d'))
    error_df['Date'] = pd.to_datetime(error_df['Date'])
    error_df = error_df.set_index('Date')

    error_df['Predicted_Close'].loc[predicted['ds']] = predicted['yhat'].iloc[-1]
    error_df['Predicted_Close_Minimum'].loc[predicted['ds']] = predicted['yhat_lower'].iloc[-1]
    error_df['Predicted_Close_Maximum'].loc[predicted['ds']] = predicted['yhat_upper'].iloc[-1]
    
    # add compnay name to the dataframe
    error_df['Company'] = company

    error_df.insert(0, 'Date', error_df.index)

    if pd.isna(real_stock_price['Date'])[0] == False:
        if predicted['ds'].iloc[-1].weekday() == 0:
            days = 3 #default days = 1
        elif predicted['ds'].iloc[-1].weekday() == 6:
            days = 2
        else:
            days = 1
            
        error_df['Actual_Close'].loc[predicted['ds']-timedelta(days)] = real_stock_price['Close'].iloc[-1]
        percent_change = ((error_df['Actual_Close'].loc[predicted['ds']-timedelta(days)] - error_df['Predicted_Close'].loc[predicted['ds']-timedelta(days)])/error_df['Actual_Close'].loc[predicted['ds']-timedelta(days)]*100)
        error_df['Percent_Change_from_Close'].loc[predicted['ds']-timedelta(days)] = percent_change

        up_or_down_original = error_df['Actual_Close'].loc[predicted['ds']][0]-error_df['Actual_Close'].loc[predicted['ds']-timedelta(days)][0]

        if up_or_down_original > 0:
            error_df['Actual_Up_Down'].loc[predicted['ds']] = 'Up'

        elif up_or_down_original == 0:
            error_df['Actual_Up_Down'].loc[predicted['ds']] = 'Same'

        else:
            error_df['Actual_Up_Down'].loc[predicted['ds']] = 'Down'


        up_or_down_predicted = error_df['Predicted_Close'].loc[predicted['ds']][0]-error_df['Predicted_Close'].loc[predicted['ds']-timedelta(days)][0]

        if up_or_down_predicted > 0:
            error_df['Predicted_Up_Down'].loc[predicted['ds']] = 'Up'

        elif up_or_down_predicted == 0:
            error_df['Predicted_Up_Down'].loc[predicted['ds']] = 'Same'

        else:
            error_df['Predicted_Up_Down'].loc[predicted['ds']] = 'Down'
        

        error_df = error_df[~error_df.index.duplicated(keep='first')]

    else:
        pass

    return error_df
  

import os
import warnings
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta, date
from dateutil.parser import parse
import json
warnings.filterwarnings("ignore")
from prophet.serialize import model_to_json, model_from_json


def inferencing(holiday_list_path, training_data_path, error_df_path, model_path):

    today = date.today()

    company_list = pd.read_csv(training_data_path)["Company"].unique()

    for company in company_list:
        
        error_df = pd.read_csv(error_df_path + company + '.csv')

        #! Checking if there were any missed days in between
        if error_df.iloc[-1]['Date'] >= str(today):
            error_df = pred_vs_real_comparision(real_stock_price(company, next_day_prediction(model_path + company + '.json', False)), next_day_prediction(model_path + company + '.json', False), error_df, company)
        else:
            print ("Missed days")
            error_df = filling_missing_dates(error_df, company, holiday_list_path, model_path)
            error_df = pred_vs_real_comparision(real_stock_price(company, next_day_prediction(model_path + company + '.json', False)), next_day_prediction(model_path + company + '.json', False), error_df, company)

        #! Check for null values in actual close and get its date
        error_df = update_actual_close(error_df, company)

        if is_holiday(today, holiday_list_path) == True or today.weekday() == 5 or today.weekday() == 6:
            error_df = error_df[error_df['Date'] != str(today)]

        # convert the dates to one format
        # error_df['Date'] = pd.to_datetime(error_df['Date'])
        # error_df['Date'] = error_df['Date'].dt.strftime('%Y-%m-%d')
        #! saving the df to a csv file
        error_df.to_csv(error_df_path + company + '.csv', index=False)


#! Loading Model
def load_model(model_path):
    with open(model_path, 'r') as fin:
        saved_model = model_from_json(json.load(fin))  # Load model
    return saved_model

#! check for holiday
def is_holiday(today, holiday_list_path):
    holidays_list = pd.read_csv(holiday_list_path)
    for i in range(len(holidays_list['Day'])):
        holidays_list['Day'][i] = pd.to_datetime(parse(holidays_list['Day'][i]))
    for i in range(len(holidays_list['Day'])):
        if holidays_list['Day'][i].date() == today:
            return True
    return False

#! get real stock price
def real_stock_price(company, predicted):

    now = datetime.now()
    weekday_weekend = datetime.strptime(str(predicted['ds'][0]), '%Y-%m-%d %H:%M:%S')
    
    if weekday_weekend.weekday() <= 5 and weekday_weekend.weekday() != 0:
        days = 1
    elif weekday_weekend.weekday() == 6:
        days = 2
    elif weekday_weekend.weekday() == 0:
        days = 3

    past = datetime.strptime(str(predicted['ds'][0]), '%Y-%m-%d %H:%M:%S') - timedelta(days)
    past = past.replace(hour = now.hour, minute = now.minute, second = now.second, microsecond = now.second)
    past = int(time.mktime(past.timetuple()))
    
    interval = '1d'

    # defining the query to get historical stock data
    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{company}?period1={past}&period2={past}&interval={interval}&events=history&includeAdjustedClose=true'
    
    try:
        company_stock_price = pd.read_csv(query_string)
        company_stock_price = company_stock_price[['Date', 'Close']]
        return company_stock_price
    except:
        company_stock_price = pd.DataFrame(np.nan, index = [0], columns=['Date'])
        return company_stock_price

#! for next day prediction
def next_day_prediction(model_path, missing_dates, missing_dates_df = 0):

    saved_model = load_model(model_path)

    if missing_dates == False:
        next_day = date.today() + timedelta(days=1)
        future_date = pd.DataFrame(pd.date_range(start = next_day, end = next_day, freq ='D'), columns = ['ds'])
        predicted = saved_model.predict(future_date)
        return (predicted[['ds','yhat', 'yhat_upper', 'yhat_lower']])
    else:
        missing_dates_df.rename(columns={'Date':'ds'}, inplace=True)
        predicted = saved_model.predict(missing_dates_df)
        return (predicted[['ds','yhat', 'yhat_upper', 'yhat_lower']])

#! Fetch stock price when there is a null value in actual price column
def fetch_stock_price(company, date):
    now = datetime.now()

    past = datetime.strptime(str(date), '%Y-%m-%d %H:%M:%S')
    past = past.replace(hour = now.hour, minute = now.minute, second = now.second, microsecond = now.second)
    past = int(time.mktime(past.timetuple()))
    
    interval = '1d'

    # defining the query to get historical stock data
    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{company}?period1={past}&period2={past}&interval={interval}&events=history&includeAdjustedClose=true'
    
    try:
        company_stock_price = pd.read_csv(query_string)
        company_stock_price = company_stock_price[['Date', 'Close']]
        return company_stock_price['Close'][0]
    except:
        return None

def update_actual_close(df, company):
    #! Check for null values in actual close and get its date
    null_values = df[df['Actual_Close'].isnull()]
    print(null_values)
    # add %H:%M:%S to get time as well
    null_values['Date'] = pd.to_datetime(null_values['Date'], format = '%Y-%m-%d %H:%M:%S')
    # convert to string
    null_values['Date'] = null_values['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    null_values = null_values['Date'].to_list()

    for date in null_values:
        stock_price = fetch_stock_price(company, date)
        # remove time from date
        date = date.split(' ')[0]
        # append to dataframe
        df.loc[df['Date'] == date, 'Actual_Close'] = stock_price
        # calculate percent change from close for the date in null_values
        df.loc[df['Date'] == date, 'Percent_Change_from_Close'] = (df.loc[df['Date'] == date, 'Predicted_Close'] - df.loc[df['Date'] == date, 'Actual_Close'])/df.loc[df['Date'] == date, 'Actual_Close']

        df['Actual_Up_Down'] = np.where(df['Actual_Close'].isna(), np.nan, np.where(df['Actual_Close'] > df['Actual_Close'].shift(1), 'Up', 'Down'))
        df['Predicted_Up_Down'] = np.where(df['Predicted_Close'].isna(), np.nan, np.where(df['Predicted_Close'] > df['Predicted_Close'].shift(1), 'Up', 'Down'))

        # df['Actual_Up_Down'] = np.where((df['Actual_Close'] > df['Actual_Close'].shift(-1)), 'Up', 'Down')
        # df['Predicted_Up_Down'] = np.where((df['Predicted_Close'] > df['Actual_Close'].shift(-1)), 'Up', 'Down')

        # convert the dates to one format
        df['Date'] = pd.to_datetime(df['Date'])
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

    return df


def real_stock_price_missing_date(company, predicted):
    now = datetime.now()
    predicted['Close'] = None
    for i in range(len(predicted['ds'])):
        past = datetime.strptime(str(predicted['ds'][i]), '%Y-%m-%d %H:%M:%S')
        past = past.replace(hour = now.hour, minute = now.minute, second = now.second, microsecond = now.second)
        print(past)
        past = int(time.mktime(past.timetuple()))
        interval = '1d'
        
        query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{company}?period1={past}&period2={past}&interval={interval}&events=history&includeAdjustedClose=true'
        company_stock_price = pd.read_csv(query_string)
        company_stock_price = company_stock_price[['Date', 'Close']]
        predicted['Close'][i] = company_stock_price['Close'].values[0]
    return predicted

#! Filling missing dates
def filling_missing_dates(error_df, company, holiday_list_path, model_path):
    Date = date.today()
    
    date_range = pd.date_range(start = error_df.iloc[-1]['Date'], end = Date, freq ='B')

    date_range_df = pd.DataFrame(columns = error_df.columns)
    date_range_df['Date'] = date_range
    date_range_df['Date'] = date_range_df['Date'].dt.date

    for i in range(len(date_range_df['Date'])):
        if is_holiday(date_range_df['Date'][i], holiday_list_path) == True:
            date_range_df = date_range_df[date_range_df['Date'] != date_range_df['Date'][i]]
            
    missing_dates_df = next_day_prediction(model_path + company + '.json',True, date_range_df)
    missing_dates_df = real_stock_price_missing_date(company, missing_dates_df)

    # convert ds from datetime to date
    missing_dates_df['ds'] = missing_dates_df['ds'].dt.date

    missing_dates_df.rename(columns = {'ds':'Date', 'Close':'Actual_Close', 'yhat':'Predicted_Close', 'yhat_upper':'Predicted_Close_Maximum', 'yhat_lower':'Predicted_Close_Minimum'}, inplace = True)
    missing_dates_df['Percent_Change_from_Close'] = ((missing_dates_df['Actual_Close'] - missing_dates_df['Predicted_Close'])/missing_dates_df['Actual_Close'])*100

    missing_dates_df['Actual_Up_Down'] = np.where((missing_dates_df['Actual_Close'] > missing_dates_df['Actual_Close'].shift(-1)), 'Up', 'Down')
    missing_dates_df['Predicted_Up_Down'] = np.where((missing_dates_df['Predicted_Close'] > missing_dates_df['Actual_Close'].shift(-1)), 'Up', 'Down')

    error_df = error_df.append(missing_dates_df, ignore_index= True)
    error_df = error_df.drop_duplicates(subset = 'Date', keep = 'last')
    error_df['Company'] = company

    return error_df

def pred_vs_real_comparision(real_stock_price, predicted, error_df, company):

    df = pd.DataFrame([[np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]], columns=error_df.columns)
    error_df = pd.concat([error_df, df], ignore_index =True)

    error_df['Date'].iloc[-1] = str(predicted['ds'].iloc[-1].strftime('%Y-%m-%d'))
    error_df['Date'] = pd.to_datetime(error_df['Date'])
    error_df = error_df.set_index('Date')

    error_df['Predicted_Close'].loc[predicted['ds']] = predicted['yhat'].iloc[-1]
    error_df['Predicted_Close_Minimum'].loc[predicted['ds']] = predicted['yhat_lower'].iloc[-1]
    error_df['Predicted_Close_Maximum'].loc[predicted['ds']] = predicted['yhat_upper'].iloc[-1]
    
    # add company name to the dataframe
    error_df['Company'] = company

    error_df.insert(0, 'Date', error_df.index)

    if pd.isna(real_stock_price['Date'])[0] == False:
        if predicted['ds'].iloc[-1].weekday() == 0:
            days = 3 #default days = 1
        elif predicted['ds'].iloc[-1].weekday() == 6:
            days = 2
        else:
            days = 1
            
        error_df['Actual_Close'].loc[predicted['ds']-timedelta(days)] = real_stock_price['Close'].iloc[-1]
        percent_change = ((error_df['Actual_Close'].loc[predicted['ds']-timedelta(days)] - error_df['Predicted_Close'].loc[predicted['ds']-timedelta(days)])/error_df['Actual_Close'].loc[predicted['ds']-timedelta(days)]*100)
        error_df['Percent_Change_from_Close'].loc[predicted['ds']-timedelta(days)] = percent_change

        up_or_down_original = error_df['Actual_Close'].loc[predicted['ds']][0]-error_df['Actual_Close'].loc[predicted['ds']-timedelta(days)][0]

        if up_or_down_original > 0:
            error_df['Actual_Up_Down'].loc[predicted['ds']] = 'Up'

        elif up_or_down_original == 0:
            error_df['Actual_Up_Down'].loc[predicted['ds']] = 'Same'

        else:
            error_df['Actual_Up_Down'].loc[predicted['ds']] = 'Down'

        up_or_down_predicted = error_df['Predicted_Close'].loc[predicted['ds']][0]-error_df['Predicted_Close'].loc[predicted['ds']-timedelta(days)][0]

        if up_or_down_predicted > 0:
            error_df['Predicted_Up_Down'].loc[predicted['ds']] = 'Up'

        elif up_or_down_predicted == 0:
            error_df['Predicted_Up_Down'].loc[predicted['ds']] = 'Same'

        else:
            error_df['Predicted_Up_Down'].loc[predicted['ds']] = 'Down'
        

        error_df = error_df[~error_df.index.duplicated(keep='first')]

    else:
        pass

    return error_df
