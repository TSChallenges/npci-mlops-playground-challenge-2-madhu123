import gradio as gr
import joblib
import numpy as np

# Step 1: Load the pre-trained model
model = joblib.load('random_forest_model.pkl')

# Step 2: Define the prediction function
def predict_bike_rent(season, hr, holiday, yr, mnth, workingday, weathersit, temp, atemp, humidity, windspeed, 
                      weekday_Fri, weekday_Mon, weekday_Sat, weekday_Sun, weekday_Thu, weekday_Tue, weekday_Wed):
    # Mapping categorical inputs to integers
    season_map = {'spring': 1, 'summer': 2, 'fall': 3, 'winter': 4}
    weathersit_map = {
        'Clear': 1, 'Few clouds': 2, 'Partly cloudy': 3, 'Mist + Cloudy': 4,
        'Mist + Broken clouds': 5, 'Mist + Few clouds': 6, 'Mist': 7, 
        'Light Snow': 8, 'Light Rain + Thunderstorm + Scattered clouds': 9, 
        'Light Rain + Scattered clouds': 10, 'Heavy Rain + Ice Pallets + Thunderstorm + Mist': 11, 'Snow + Fog': 12
    }
    
    # Convert categorical inputs
    season = season_map[season]
    weathersit = weathersit_map[weathersit]
    holiday = int(holiday)
    workingday = int(workingday)

    # Convert weekday checkboxes to binary
    weekday_Fri = int(weekday_Fri)
    weekday_Mon = int(weekday_Mon)
    weekday_Sat = int(weekday_Sat)
    weekday_Sun = int(weekday_Sun)
    weekday_Thu = int(weekday_Thu)
    weekday_Tue = int(weekday_Tue)
    weekday_Wed = int(weekday_Wed)
    
    # Prepare input features
    input_features = np.array([[season, hr, holiday, yr, mnth, workingday, weathersit, temp, atemp, humidity, windspeed, 
                                weekday_Fri, weekday_Mon, weekday_Sat, weekday_Sun, weekday_Thu, weekday_Tue, weekday_Wed]])
    
    # Predict using the model
    prediction = model.predict(input_features)
    return f"Predicted bike rentals: {int(prediction[0])}"

# Gradio Interface
interface = gr.Interface(
    fn=predict_bike_rent,
    inputs=[
        gr.Dropdown(choices=['spring', 'summer', 'fall', 'winter'], label="Season"),
        gr.Slider(minimum=0, maximum=23, step=1, label="Hour of the Day (hr)"),
        gr.Checkbox(label="Is Holiday?"),
        gr.Slider(minimum=0, maximum=1, step=1, label="Year (yr) (0 = 2011, 1 = 2012)", value=1),
        gr.Slider(minimum=1, maximum=12, step=1, label="Month (mnth)"),
        gr.Checkbox(label="Is Working Day?"),
        gr.Dropdown(choices=['Clear', 'Few clouds', 'Partly cloudy', 'Mist + Cloudy', 'Mist + Broken clouds', 'Mist + Few clouds', 
                             'Mist', 'Light Snow', 'Light Rain + Thunderstorm + Scattered clouds', 'Light Rain + Scattered clouds', 
                             'Heavy Rain + Ice Pallets + Thunderstorm + Mist', 'Snow + Fog'], label="Weather Situation"),
        gr.Slider(minimum=-10, maximum=40, step=1, label="Temperature (°C)"),
        gr.Slider(minimum=-10, maximum=40, step=1, label="Feels Like Temperature (°C)"),
        gr.Slider(minimum=0, maximum=100, step=1, label="Humidity (%)"),
        gr.Slider(minimum=0, maximum=100, step=1, label="Windspeed (km/h)"),
        gr.Checkbox(label="Is Friday?"),
        gr.Checkbox(label="Is Monday?"),
        gr.Checkbox(label="Is Saturday?"),
        gr.Checkbox(label="Is Sunday?"),
        gr.Checkbox(label="Is Thursday?"),
        gr.Checkbox(label="Is Tuesday?"),
        gr.Checkbox(label="Is Wednesday?")
    ],
    outputs="text",
    title="Bike Rental Prediction",
    description="Enter the features to predict the number of bike rentals, including season, hour, weather, and weekday information."
)

# Launch the interface
interface.launch()