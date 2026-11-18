import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create dataset
X = np.array([[1], [2], [3], [4], [5],
              [6], [7], [8], [9], [10]])

Y = np.array([2, 4, 5, 4, 5, 7, 8, 9, 10, 12])

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, Y_train)

# Predict values
Y_pred = model.predict(X_test)

# Calculate performance
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

# Display results
print("Actual Values:", Y_test)
print("Predicted Values:", Y_pred)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Plot regression line
plt.scatter(X, Y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red',
         label='Regression Line')

plt.xlabel("Independent Variable (X)")
plt.ylabel("Dependent Variable (Y)")
plt.title("Linear Regression")
plt.legend()
plt.show()
