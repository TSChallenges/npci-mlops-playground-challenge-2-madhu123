import re
import joblib
import gradio
import gradio as gr
# Loading the Required Packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Load model
trained_model = joblib.load(filename = "random_forest_model.pkl")


# Prediction function
#model_rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
# Fit the model
#model_rf.fit(X_train_scaled, y_train.values.ravel())

# Prediction for test set
#y_pred = model_rf.predict(x_test_scaled)

# Calculate the score/error
#print("R2 score:", r2_score(y_test, y_pred))
#print("Mean squared error:", mean_squared_error(y_test, y_pred))

# UI - Input components
in_tdedate = gradio.Textbox(lines=1, placeholder=None, value="2011-June-24", label='hourly date')
in_season = gradio.Radio(["spring", "summer", "fall", "winter"], type="value", label='season')
in_hr = gradio.Textbox(lines=1, placeholder=None, value="1am", label='hr')
in_holiday = gradio.Radio(["Yes", "No"], type="value", label='is holiday')
in_weekday = gradio.Radio(["Sat", "sun"], type="value", label=' weekday')
in_workingday = gradio.Radio(["mon", "tue", "wed", "thu", "fri"], type="value", label='workingday')
in_weather = gradio.Radio(["Heavy Rain", "Light Rain", "Mist", "Clear"], type="value", label='weatherist')
in_temp = gradio.Textbox(lines=1, placeholder=None, value="10", label='temperature in celsius')
in_atemp = gradio.Textbox(lines=1, placeholder=None, value="0", label='temperature in celsius')
in_hum = gradio.Textbox(lines=1, placeholder=None, value="40", label='humidity')
in_windspeed = gradio.Textbox(lines=1, placeholder=None, value="16.99", label='windspeed')

# UI - Output component
out_label = gradio.Textbox(type="text", label='Prediction', elem_id="out_textbox")

# Mappings for categorical features
season_mapping = {'spring': 0, 'winter': 1, 'summer': 2, 'fall': 3}
yr_mapping = {2011: 0, 2012: 1}
mnth_mapping = {'January': 0, 'February': 1, 'December': 2, 'March': 3, 'November': 4, 'April': 5,
                'October': 6, 'May': 7, 'September': 8, 'June': 9, 'July': 10, 'August': 11}
weather_mapping = {'Heavy Rain': 0, 'Light Rain': 1, 'Mist': 2, 'Clear': 3}
holiday_mapping = {'Yes': 0, 'No': 1}
workingday_mapping = {'No': 0, 'Yes': 1}
hour_mapping = {'4am': 0, '3am': 1, '5am': 2, '2am': 3, '1am': 4, '12am': 5, '6am': 6, '11pm': 7, '10pm': 8,
                '10am': 9, '9pm': 10, '11am': 11, '7am': 12, '9am': 13, '8pm': 14, '2pm': 15, '1pm': 16,
                '12pm': 17, '3pm': 18, '4pm': 19, '7pm': 20, '8am': 21, '6pm': 22, '5pm': 23}



# Label prediction function
def get_output_label(in_tdedate, in_season, in_hr, in_holiday, in_weekday, in_workingday, in_weather, in_temp, in_atemp, in_hum, in_windspeed):
    
    input_df = pd.DataFrame({'tdedate': [in_tdedate], 
                             'season': [season_mapping[in_season]], 
                             'hour': [hour_mapping[in_hr]],
                             'holiday': [holiday_mapping[in_holiday]],
                             'weekday': [in_weekday],
                             'workingday': [workingday_mapping[in_workingday]],
                             #'yr': [yr_mapping[in_yr]],
                             'weather': [weather_mapping[in_weather]],
                             'temp': [in_temp],
                             'atemp': [in_atemp],
                             'hum': [in_hum],
                             'windspeed': [in_windspeed]})
    
    prediction = trained_model.predict(input_df)     # Make a prediction using the saved model
    print(prediction[0])
#    if prediction[0] == 1:
#        label = "Likely to Survive"
#    else:
#        label = "Less likely to Survive"

    return (prediction[0])


# Create Gradio interface object
iface = gradio.Interface(fn = get_output_label,
                         inputs = [in_tdedate, in_season, in_hr, in_holiday, in_weekday, in_workingday, in_weather, in_temp, in_atemp, in_hum, in_windspeed],
                         outputs = [out_label],
                         title="bike rental app",
                         description="bike rent hours model",
                         flagging_mode='never'
                         )

# Launch gradio interface
iface.launch(server_name = "0.0.0.0", server_port = 7860)
                         # set server_name = "0.0.0.0" and server_port = 7860 while launching it inside container.
                         # default server_name = "127.0.0.1", and server_port = 7860