import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

cab_rides_and_weather = pd.read_csv('sample_cab_rides_and_weather.csv')

le = LabelEncoder()
cab_rides_and_weather['name_encoded'] = le.fit_transform(cab_rides_and_weather['name'])
cab_rides_and_weather['source_encoded'] = le.fit_transform(cab_rides_and_weather['source'])
cab_rides_and_weather['destination_encoded'] = le.fit_transform(cab_rides_and_weather['destination'])

X = cab_rides_and_weather[['distance', 'temp', 'rain', 'wind', 'surge_multiplier', 'name_encoded', 'source_encoded', 'destination_encoded']]
y = cab_rides_and_weather['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

SVRmodel = SVR(kernel='rbf', C=10, epsilon=0.5, gamma='scale')
SVRmodel.fit(X_train, y_train)
SVRModelScore = SVRmodel.score(X_test, y_test)

print(f"SVR Model Score: {SVRModelScore}")
