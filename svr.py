import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

cab_rides_and_weather = pd.read_csv('sample_cab_rides_and_weather.csv')

cab_rides_and_weather = pd.get_dummies(
    cab_rides_and_weather,
    columns=['name', 'source', 'destination']
)

X = cab_rides_and_weather.drop(columns=['price', 'surge_multiplier'])
y = cab_rides_and_weather['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

SVRModel = SVR(kernel='rbf', C=10, epsilon=0.5, gamma='scale')
SVRModel.fit(X_train, y_train)
SVRModelScore = SVRModel.score(X_test, y_test)

print(f"SVR Model Score: {SVRModelScore:0.4f}")
