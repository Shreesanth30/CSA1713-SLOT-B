
# Decision Tree Implementation

# Dataset: [Hours Studied, Result]
# Result: 0 = Fail, 1 = Pass

data = [
    [1, 0],
    [2, 0],
    [3, 0],
    [4, 1],
    [5, 1],
    [6, 1]
]

# Training the decision tree using a simple rule
def decision_tree(hours):
    if hours <= 3:
        return "Fail"
    else:
        return "Pass"

# Display the dataset
print("Hours Studied\tResult")
print("-------------------------")

for row in data:
    result = "Pass" if row[1] == 1 else "Fail"
    print(row[0], "\t\t", result)

# Test the decision tree
print("\nDecision Tree Predictions")
print("-------------------------")

hours = int(input("Enter hours studied: "))

prediction = decision_tree(hours)

print("Prediction:", prediction)

