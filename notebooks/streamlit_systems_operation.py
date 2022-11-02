import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from functions_for_streamlit import *
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

holiday_list_path = 'https://raw.githubusercontent.com/advait-t/Stock-Prediction/master/data/raw/2017-2022_Holidays_NSE_BSE_EQ_EQD.csv'
training_data_path = 'https://raw.githubusercontent.com/advait-t/Stock-Prediction/master/data/raw/training_data.csv'
error_df_path = 'https://raw.githubusercontent.com/advait-t/Stock-Prediction/master/data/Final/error_df1'
model_path = 'https://raw.githubusercontent.com/advait-t/Stock-Prediction/master/models/Model_JSON/'
companies_list_path = 'https://raw.githubusercontent.com/advait-t/Stock-Prediction/master/config/process/companies_config.txt'
error_metrics_path  = 'https://raw.githubusercontent.com/advait-t/Stock-Prediction/master/models/Model_Metrics/error_metrics.csv'


st.set_page_config(layout="wide")

st.sidebar.title('Daily Stock Price Prediction')

today = date.today()
yesterday = today - timedelta(days=1)

#! select box for choosing the company
company = st.sidebar.selectbox('Select the company', pd.read_csv(training_data_path)["Company"].unique())

error_df = pd.read_csv(error_df_path + company + '.csv')

st.title(company.replace('.NS', ''))

#! filtering the dataframe as per the company selected
error_df = error_df[error_df['Company'] == company]

error_df = error_df.set_index('Date', drop = False)
error_df['Date'] = error_df.index

#! making 3 columns for
col1, col2, col3 = st.columns((1,1,1))

#! for col1 or yesterday's prediction
yesterday = datetime.now().date() - timedelta(1)

if yesterday.weekday() == 5 or yesterday.weekday() == 6 or is_holiday(yesterday, holiday_list_path) == True:
    yesterday_date = date.today() - timedelta(days = 1)
    yesterday_date = str(yesterday_date.strftime('%d-%B-%Y'))
    col1.header('%s'%yesterday_date)
    col1.markdown('**Market Holiday**')

else:
    yesterday_date = str(datetime.now().date() - timedelta(1))

    yesterday_close, yesterday_predicted_close, yesterday_actual_up_down, yesterday_predicted_up_down = print_metrics_columns(error_df, yesterday_date)

    col1.header('%s'%(str((date.today() - timedelta(days=1)).strftime('%d-%B-%Y'))))
    col1.metric('Actual Price', '\u20B9' + str(yesterday_close), yesterday_actual_up_down)
    col1.metric('Predicted Price', '\u20B9' + str(yesterday_predicted_close), yesterday_predicted_up_down)

#! for col2 or today's prediction
todays_date = str(datetime.now().date())

if datetime.now().weekday() == 5 or datetime.now().weekday() == 6 or is_holiday(datetime.now().date(), holiday_list_path) == True:
    col2.header('Today')
    col2.markdown('**Market Holiday**')

else:
    todays_close, todays_predicted_close, todays_actual_up_down, todays_predicted_up_down = print_metrics_columns(error_df, todays_date)

    col2.header('Today')
    if datetime.now().time() > datetime.strptime('15:31', '%H:%M').time():
        col2.metric('Actual Price', '\u20B9' + str(todays_close), todays_actual_up_down)
    col2.metric('Predicted Price', '\u20B9' + str(todays_predicted_close), todays_predicted_up_down)


#! for col3 or tomorrow's prediction
if datetime.today().weekday() not in [5, 6] or is_holiday(today, holiday_list_path) == False:
    tomorrows_date = str(datetime.now().date() + timedelta(1))
    tomorrows_predicted_close = (error_df[error_df['Date'] == tomorrows_date]['Predicted_Close'].values[0]).round(2)
    tomorrows_predicted_up_down = error_df[error_df['Date'] == tomorrows_date]['Predicted_Up_Down'].values[0]
    if tomorrows_predicted_up_down == 'Up':
        tomorrows_predicted_up_down =  '+Up'
    elif tomorrows_predicted_up_down == 'Down':
        tomorrows_predicted_up_down = '-Down'

    col3.header('%s'%(str((date.today() + timedelta(days=1)).strftime('%d-%B-%Y'))))
    col3.metric('Predicted Price','\u20B9' + str(tomorrows_predicted_close), tomorrows_predicted_up_down)

else:
    if datetime.today().weekday() == 5:
        monday_date = date.today() + timedelta(days=2)
    elif datetime.today().weekday() != 5 or is_holiday == True:
        monday_date = date.today() + timedelta(days=1)

    monday_date = str(monday_date.strftime('%d-%B-%Y')) # type: ignore
    mondays_predicted_close = (error_df[error_df['Date'] == monday_date]['Predicted_Close'].values[0]).round(2)
    mondays_predicted_up_down = error_df[error_df['Date'] == monday_date]['Predicted_Up_Down'].values[0]
    if mondays_predicted_up_down == 'Up':
        tomorrows_predicted_up_down =  '+Up'
    elif mondays_predicted_up_down == 'Down':
        tomorrows_predicted_up_down = '-Down'

    col3.header('%s'%monday_date)
    col3.metric('Predicted Price','\u20B9' + str(mondays_predicted_close), tomorrows_predicted_up_down) # type: ignore

#! printing the next prediction day and time
if datetime.now().weekday() == 5:
    one_day_later_date = datetime.now() + timedelta(days=2)
elif datetime.now().weekday() == 6 or is_holiday(today, holiday_list_path) == True:
    one_day_later_date = datetime.now() + timedelta(days=1)
else:
    if datetime.now().hour < 15 :
        one_day_later_date = (datetime.now())
    else:
        one_day_later_date = (datetime.now()) + timedelta(1)

one_day_later_date = one_day_later_date.replace(hour = 15, minute = 31, second = 00, microsecond = 00)
one_day_later_date = one_day_later_date.strftime('%d-%B-%Y @ %H:%M:%S')

st.sidebar.header('Next Prediction:')
st.sidebar.success(one_day_later_date)

#! line chart for actual and predicted values
st.subheader('Actual vs Predicted Prices')

# error_df['Date'] = pd.to_datetime(error_df['Date'])
# error_df['Date'] = error_df['Date'].dt.strftime('%d-%b-%Y')
plot = px.line(error_df, x='Date', y=['Actual_Close', 'Predicted_Close'], title='Actual vs Predicted Prices', markers = True)
st.plotly_chart(plot, use_container_width=True)

#! writing the accuracy of the model
new_accuracy = compute_accuracy(error_df['Actual_Up_Down'].values, error_df['Predicted_Up_Down'].values)
st.sidebar.header('Accuracy of Predicting Up/Down:')
st.sidebar.success(f"{new_accuracy}%")
