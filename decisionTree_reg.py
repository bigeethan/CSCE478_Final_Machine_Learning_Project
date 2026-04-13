import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

cab_rides_and_weather = pd.read_csv('sample_cab_rides_and_weather.csv')
cab_rides_and_weather = pd.get_dummies(
    cab_rides_and_weather,
    columns=['name', 'source', 'destination']
)

X = cab_rides_and_weather.drop(columns=['price', 'surge_multiplier'])
y = cab_rides_and_weather['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

DTRModel = DecisionTreeRegressor(max_depth=10, random_state=42)
DTRModel.fit(X_train, y_train)
DTRModelScore = DTRModel.score(X_test, y_test)

print(f"Decision Tree with Regression Test Score: {DTRModelScore:0.4f}")
