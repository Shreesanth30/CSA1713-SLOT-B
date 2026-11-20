from sklearn import svm

# Training data
X = [[1, 2], [2, 3], [3, 3],
     [6, 7], [7, 8], [8, 8]]

# Class labels
y = [0, 0, 0, 1, 1, 1]

# Create SVM model
model = svm.SVC(kernel='linear')

# Train the model
model.fit(X, y)

# Test data
test_data = [[2, 2], [7, 7], [3, 4], [8, 9]]

# Predict the classes
predictions = model.predict(test_data)

print("SVM Binary Classification Results")

for i in range(len(test_data)):
    print("Input:", test_data[i],
          "Predicted Class:", predictions[i])
