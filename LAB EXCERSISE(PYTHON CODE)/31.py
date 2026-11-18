from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()

X = iris.data
Y = iris.target

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train, Y_train)

# Predict the output
Y_pred = knn.predict(X_test)

# Display results
print("Actual Values:", Y_test)
print("Predicted Values:", Y_pred)

# Calculate accuracy
accuracy = accuracy_score(Y_test, Y_pred)

print("KNN Classification Accuracy:", accuracy * 100, "%")
