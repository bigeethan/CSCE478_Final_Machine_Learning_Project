import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

cab_rides_and_weather = pd.read_csv('sample_cab_rides_and_weather.csv')
le_name = LabelEncoder()
le_source = LabelEncoder()
le_destination = LabelEncoder()

cab_rides_and_weather['name_encoded'] = le_name.fit_transform(cab_rides_and_weather['name'])
cab_rides_and_weather['source_encoded'] = le_source.fit_transform(cab_rides_and_weather['source'])
cab_rides_and_weather['destination_encoded'] = le_destination.fit_transform(cab_rides_and_weather['destination'])


X = cab_rides_and_weather[['distance', 
                           'temp', 
                           'rain', 
                           'wind', 
                           'surge_multiplier',
                           'name_encoded',
                           'source_encoded',
                           'destination_encoded'
                            ]]
y = cab_rides_and_weather['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

DTR = DecisionTreeRegressor(max_depth=10, random_state=43)
DTR.fit(X_train, y_train)
score = DTR.score(X_test, y_test)
print(f"Test Score: {score:0.4f}")

LR = LinearRegression()
LR.fit(X_train, y_train)
predict = LR.predict(X_test)
score_lr = metrics.r2_score(y_test, predict)
print(f"Linear Regression Test Score: {score_lr:0.4f}")



