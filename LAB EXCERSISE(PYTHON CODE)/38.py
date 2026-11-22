from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
iris = datasets.load_iris()

# Select two classes for binary classification
X = iris.data[iris.target != 2]
y = iris.target[iris.target != 2]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create SVM model
model = SVC(kernel='linear')

# Train the model
model.fit(X_train, y_train)

# Predict results
y_pred = model.predict(X_test)

# Display output
print("Actual Values:", y_test)
print("Predicted Values:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")
