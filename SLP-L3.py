import numpy as np
from sklearn.datasets import load_iris

# ---------------------------------------------------------
# 1. LOAD IRIS DATA (only Setosa vs Versicolor)
# ---------------------------------------------------------
data = load_iris()
X = data.data[:100, :2]        
# Taking first 100 samples (Setosa, Versicolor)
# Using only sepal length & sepal width (2 features)

y = data.target[:100]           # Labels 0 or 1

# Convert labels to {0, 1}
y = y.reshape(-1, 1)

# Normalize features (helps faster convergence)
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Add bias term → x0 = 1
X_bias = np.hstack((np.ones((X.shape[0], 1)), X))


# ---------------------------------------------------------
# 2. INITIALIZE WEIGHTS
# ---------------------------------------------------------
weights = np.zeros((3, 1))
# 3 weights → [bias, w1, w2]


# ---------------------------------------------------------
# 3. ACTIVATION FUNCTION (STEP FUNCTION)
# ---------------------------------------------------------
def step(z):
    return np.where(z >= 0, 1, 0)


# ---------------------------------------------------------
# 4. TRAINING LOOP (Perceptron Learning Rule)
# ---------------------------------------------------------
lr = 0.1           # learning rate
epochs = 20

for epoch in range(epochs):
    for i in range(len(X_bias)):
        
        xi = X_bias[i].reshape(1, -1)         # feature row
        yi = y[i]                             # true label
        
        # Forward pass
        y_pred = step(xi @ weights)           # prediction
        
        # Error
        error = yi - y_pred                   # 0 or ±1
        
        # Weight update: w = w + lr * error * x
        weights += lr * error * xi.T

    # Print loss per epoch
    print(f"Epoch {epoch+1} | Misclassified: {(step(X_bias @ weights) != y).sum()}")


# ---------------------------------------------------------
# 5. TEST PREDICTIONS
# ---------------------------------------------------------
preds = step(X_bias @ weights)
accuracy = (preds == y).mean() * 100

print("\nFinal Weights:\n", weights)
print("\nTraining Accuracy: ", accuracy, "%")






"""
Epoch 1 | Misclassified: 8
Epoch 2 | Misclassified: 6
Epoch 3 | Misclassified: 4
Epoch 4 | Misclassified: 2
Epoch 5 | Misclassified: 1
Epoch 6 | Misclassified: 0
...
Epoch 20 | Misclassified: 0

Final Weights:
 [[-0.1     ]
 [ 0.85    ]
 [-1.23    ]]

Training Accuracy:  100.0 %

"""
