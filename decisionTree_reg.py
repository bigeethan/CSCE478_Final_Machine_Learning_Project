import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

cab_rides_and_weather = pd.read_csv('sample_cab_rides_and_weather.csv')
le = LabelEncoder()

cab_rides_and_weather['name_encoded'] = le.fit_transform(cab_rides_and_weather['name'])
cab_rides_and_weather['source_encoded'] = le.fit_transform(cab_rides_and_weather['source'])
cab_rides_and_weather['destination_encoded'] = le.fit_transform(cab_rides_and_weather['destination'])

X = cab_rides_and_weather[['distance', 
                           'temp', 
                           'rain', 
                           'wind', 
                           'surge_multiplier',
                           'name_encoded',
                           'source_encoded',
                           'destination_encoded']]
y = cab_rides_and_weather['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

DTRModel = DecisionTreeRegressor(max_depth=10, random_state=42)
DTRModel.fit(X_train, y_train)
DTRModelScore = DTRModel.score(X_test, y_test)

print(f"DTR Model Score: {DTRModelScore:0.4f}")


#Confidence Interval
pred_test = DTRModel.predict(X_test)
errors = np.abs(y_test.values - pred_test)
mean_error = np.mean(errors)
mae = metrics.mean_absolute_error(y_test, pred_test)
var_error = np.var(errors, ddof=1)
std_error = np.std(errors, ddof=1)

#MAE = 1.96 * variance / sqrt(number of samples)
n = len(errors)
z = 1.96
ci_lower = mean_error - z * (std_error / np.sqrt(n))
ci_upper = mean_error + z * (std_error/ np.sqrt(n))


# print(f"SVR MAE: {mae:0.4f}")
print(f"DTR Mean Error : {mean_error:0.4f}")
print(f"DTR Variance Error: {var_error:0.4f}")
print(f"DTR 95% CI for MAE: [{ci_lower:0.4f}, {ci_upper:0.4f}]")