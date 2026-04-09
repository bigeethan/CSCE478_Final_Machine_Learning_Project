import kagglehub
import pandas as pd
import os

# Download latest version
path = kagglehub.dataset_download("ravi72munde/uber-lyft-cab-prices")

print("Path to dataset files:", path)

CAB_PATH = os.path.join(path, "cab_rides.csv")
WEATHER_PATH = os.path.join(path, "weather.csv")

cab_rides = pd.read_csv(CAB_PATH)
weather = pd.read_csv(WEATHER_PATH)

cab_rides['time_hour'] = (cab_rides['time_stamp'] // 1000 // 3600) * 3600
weather['time_hour'] = (weather['time_stamp'] // 3600) * 3600
weather_aggregate = weather.groupby(['location', 'time_hour']).mean(numeric_only=True).reset_index()

merged_dataset = pd.merge(
    cab_rides,
    weather_aggregate,
    left_on=['source', 'time_hour'],
    right_on=['location', 'time_hour'],
    how='left'
)

# merged_dataset.to_csv('cab_rides_and_weather.csv', index=False) <-- unnecessary to generate - just keep is a dataframe

cab_rides_and_weather = merged_dataset
sample_cab_rides_and_weather = cab_rides_and_weather.sample(n=250000, random_state=42)
data = sample_cab_rides_and_weather[['distance', 'temp', 'rain', 'wind', 'surge_multiplier', 'name', 'source', 'destination', 'price']].dropna()
data.to_csv('sample_cab_rides_and_weather.csv', index=False)