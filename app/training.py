import pandas as pd

train_df = pd.read_csv('train.csv')

test_df = pd.read_csv('test.csv')

import numpy as np
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two points specified by their latitude and longitude.

    Parameters:
    - lat1, lon1: Latitude and longitude of the first point.
    - lat2, lon2: Latitude and longitude of the second point.

    Returns:
    - Distance in kilometers.
    """
    R = 6371  # Earth radius in kilometers

    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

    # Calculate differences
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    # Calculate distance
    distance = R * c
    return distance

train_df["distance"] = haversine(train_df["pickup_latitude"],train_df["pickup_longitude"],train_df["dropoff_latitude"],train_df["dropoff_longitude"])

# Drop unecessary columns
train_df.drop(columns=['id', 'dropoff_datetime'], inplace=True)

# Transform the types into datetime
train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime'])

# Creating hierarchy in the datetime columns
train_df['pickup_year'] = train_df['pickup_datetime'].dt.year
train_df['pickup_month'] = train_df['pickup_datetime'].dt.month
train_df['pickup_day'] = train_df['pickup_datetime'].dt.day
train_df['pickup_day_of_week'] = train_df['pickup_datetime'].dt.day_of_week
train_df['pickup_hours'] = train_df['pickup_datetime'].dt.hour

train_df.drop(columns=['pickup_datetime'], inplace=True)

# Normalize the store and fwd flag
train_df = pd.get_dummies(train_df,columns=['store_and_fwd_flag'],drop_first=True,dtype=int)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Assuming train_df is your DataFrame
# Drop the target variable from X and select it for y
X = train_df.drop(columns=['trip_duration'])
y = train_df['trip_duration']


# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the selected columns
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Regressor model
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data
rf_regressor.fit(X_train, y_train)

# Make predictions on the test data
predictions = rf_regressor.predict(X_test)

# Add 1 to predictions and actual values before taking the logarithm 
log_predictions = np.log(predictions)
log_y_test = np.log(y_test)

# Calculate squared diffrences
squared_log_diff = (log_predictions - log_y_test) ** 2

# Calculate the mean squared logarithmic error
mean_squared_log_error = np.mean(squared_log_diff)

# Calculate the root mean squared logarithmic error
rmsle = np.sqrt(mean_squared_log_error)

# Evaluate the model
print(f'Mean Squared Error: {rmsle}')

test_df["distance"] = haversine(test_df["pickup_latitude"],test_df["pickup_longitude"],test_df["dropoff_latitude"],test_df["dropoff_longitude"])

# Transform the types into datetime
test_df['pickup_datetime'] = pd.to_datetime(test_df['pickup_datetime'])

# Creating hierarchy in the datetime columns
test_df['pickup_year'] = test_df['pickup_datetime'].dt.year
test_df['pickup_month'] = test_df['pickup_datetime'].dt.month
test_df['pickup_day'] = test_df['pickup_datetime'].dt.day
test_df['pickup_day_of_week'] = test_df['pickup_datetime'].dt.day_of_week
test_df['pickup_hours'] = test_df['pickup_datetime'].dt.hour

test_df = pd.get_dummies(test_df,columns=['store_and_fwd_flag'],drop_first=True,dtype=int)
test_df.drop(columns=['id', 'pickup_datetime'], inplace=True)

# Drop any columns from test_df that are not in X_train (features used during training)
test_data = test_df[train_df.drop(columns=['trip_duration']).columns]

# Fit and transform the selected columns
test_data = scaler.fit_transform(test_data)

# Make predictions on the test data
predictions_test = rf_regressor.predict(test_data)

predicted_df = test_df
predicted_df['trip_duration'] = predictions_test

predicted_df.to_csv('predicted.csv')

print('Processing done')