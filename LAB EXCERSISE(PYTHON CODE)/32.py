from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()

X = iris.data
Y = iris.target

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Create Decision Tree model
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X_train, Y_train)

# Predict output
Y_pred = model.predict(X_test)

# Display results
print("Actual Values:", Y_test)
print("Predicted Values:", Y_pred)

# Calculate accuracy
accuracy = accuracy_score(Y_test, Y_pred)

print("Decision Tree Accuracy:", accuracy * 100, "%")
