import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset, Download the dataset through :
# https://www.kaggle.com/datasets/rockysai/house-price-prediction
data = pd.read_csv('House Prediction.csv')

# Define features and target variable
X = data[['square_feet', 'bedrooms', 'bathrooms']]
y = data['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Ask for new house details from the user
square_feet = float(input("Enter square footage of the new house: "))
bedrooms = int(input("Enter number of bedrooms in the new house: "))
bathrooms = int(input("Enter number of bathrooms in the new house: "))

# Make prediction for the new house
new_house = [[square_feet, bedrooms, bathrooms]]
predicted_price = model.predict(new_house)
print(f'Predicted price for the new house: {predicted_price[0]}')
