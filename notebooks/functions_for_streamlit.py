import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import plotly.express as px
from dateutil.parser import parse
import warnings
warnings.filterwarnings("ignore")

#! check for holiday
def is_holiday(today, holidays_list_path):
    holidays_list = pd.read_csv(holidays_list_path)
    for i in range(len(holidays_list['Day'])):
        holidays_list['Day'][i] = pd.to_datetime(parse(holidays_list['Day'][i]))
    for i in range(len(holidays_list['Day'])):
        if holidays_list['Day'][i].date() == today:
            return True
    return False

def compute_accuracy(y_true, y_pred):
    correct_predictions = 0
    # iterate over each label and check
    for true, predicted in zip(y_true, y_pred):
        if true == predicted:
            correct_predictions += 1
    # compute the accuracy
    accuracy = round(((correct_predictions/len(y_true))*100),2)

    return accuracy

def print_metrics_columns(error_df, Date):
    actual_close = (error_df[error_df['Date'] == Date]['Actual_Close'].values[0]).round(2)
    predicted_close = (error_df[error_df['Date'] == Date]['Predicted_Close'].values[0]).round(2)
    actual_up_down = error_df[error_df['Date'] == Date]['Actual_Up_Down'].values[0]
    predicted_up_down = error_df[error_df['Date'] == Date]['Predicted_Up_Down'].values[0]
    if predicted_up_down == 'Up':
        predicted_up_down =  '+Up'
    elif predicted_up_down == 'Down':
        predicted_up_down = '-Down'
    
    if actual_up_down == 'Up':
        actual_up_down =  '+Up'
    elif actual_up_down == 'Down':
        actual_up_down = '-Down'

    return actual_close, predicted_close, actual_up_down, predicted_up_down