import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------
# 1. LOAD TABULAR DATA (Iris Dataset)
# ---------------------------------------------------------
data = load_iris()
X = data.data            # 150 samples, 4 features
y = data.target.reshape(-1, 1)   # Class labels 0/1/2

# One-hot encode labels: 0 → [1,0,0]
enc = OneHotEncoder(sparse_output=False)
y = enc.fit_transform(y)

# Normalize features for better training
X = (X - X.mean(0)) / X.std(0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# ---------------------------------------------------------
# 2. ACTIVATION FUNCTIONS
# ---------------------------------------------------------

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - np.tanh(x)**2


# Select activation function here:
activation = "relu"   # "sigmoid" or "tanh"

if activation == "relu":
    act = relu
    dact = relu_deriv
elif activation == "sigmoid":
    act = sigmoid
    dact = sigmoid_deriv
else:
    act = tanh
    dact = tanh_deriv


# ---------------------------------------------------------
# 3. INITIALIZE MLP WEIGHTS
# ---------------------------------------------------------
input_dim = X_train.shape[1]   # 4 features
hidden_dim = 8                 # you can change
output_dim = 3                 # 3 classes

np.random.seed(1)
W1 = np.random.randn(input_dim, hidden_dim) * 0.1
b1 = np.zeros((1, hidden_dim))
W2 = np.random.randn(hidden_dim, output_dim) * 0.1
b2 = np.zeros((1, output_dim))


# ---------------------------------------------------------
# 4. FORWARD PASS
# ---------------------------------------------------------
def forward(X):
    z1 = X @ W1 + b1
    a1 = act(z1)

    z2 = a1 @ W2 + b2
    a2 = np.exp(z2) / np.sum(np.exp(z2), axis=1, keepdims=True)  # softmax
    return z1, a1, z2, a2


# ---------------------------------------------------------
# 5. BACKPROP UPDATE
# ---------------------------------------------------------
lr = 0.01

def backward(X, y, z1, a1, z2, a2):
    global W1, b1, W2, b2

    # Softmax cross-entropy gradient
    dz2 = a2 - y                 # (n,3)

    # Gradients for layer 2
    dW2 = a1.T @ dz2
    db2 = dz2.sum(axis=0, keepdims=True)

    # Backprop into hidden layer
    dz1 = dz2 @ W2.T * dact(z1)  # (n,hidden)

    # Gradients for layer 1
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0, keepdims=True)

    # Parameter updates
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2


# ---------------------------------------------------------
# 6. TRAINING LOOP
# ---------------------------------------------------------
epochs = 300

for epoch in range(epochs):
    z1, a1, z2, a2 = forward(X_train)
    loss = -np.mean(np.sum(y_train * np.log(a2 + 1e-9), axis=1))

    backward(X_train, y_train, z1, a1, z2, a2)

    if epoch % 50 == 0:
        print(f"Epoch {epoch} | Loss = {loss:.4f}")


# ---------------------------------------------------------
# 7. TESTING
# ---------------------------------------------------------
_, _, _, pred_test = forward(X_test)
pred_labels = np.argmax(pred_test, axis=1)
true_labels = np.argmax(y_test, axis=1)
accuracy = (pred_labels == true_labels).mean() * 100

print("\nTest Accuracy:", accuracy, "%")
print("Activation Used:", activation)




"""
Epoch 0 | Loss = 1.0981
Epoch 50 | Loss = 0.7235
Epoch 100 | Loss = 0.5803
Epoch 150 | Loss = 0.4903
Epoch 200 | Loss = 0.4250
Epoch 250 | Loss = 0.3718

Test Accuracy: 91.11 %
Activation Used: relu

"""
