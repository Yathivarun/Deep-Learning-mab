import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# ------------------------------------------------------------
# 1. LOAD 32x32 IMAGE DATASET (CIFAR-10)
# ------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values (0-255 → 0-1)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Flatten labels
y_train = y_train.flatten()
y_test = y_test.flatten()

print("Training samples:", x_train.shape)
print("Testing samples:", x_test.shape)


# ------------------------------------------------------------
# 2. DESIGN MLP MODEL
# ------------------------------------------------------------
model = models.Sequential([
    
    layers.Flatten(input_shape=(32, 32, 3)),   # Convert 32x32x3 → 3072

    layers.Dense(256, activation="relu"),      # Hidden layer 1
    layers.Dense(128, activation="relu"),      # Hidden layer 2

    layers.Dense(10, activation="softmax")     # Output layer (10 classes)
])

model.summary()


# ------------------------------------------------------------
# 3. COMPILE MODEL
# ------------------------------------------------------------
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ------------------------------------------------------------
# 4. TRAIN MODEL
# ------------------------------------------------------------
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    verbose=2
)


# ------------------------------------------------------------
# 5. TEST ACCURACY
# ------------------------------------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("\nTest Accuracy:", test_acc)





"""
Training samples: (50000, 32, 32, 3)
Testing samples: (10000, 32, 32, 3)

Epoch 1: val_accuracy ≈ 40–45%
Epoch 5: val_accuracy ≈ 48–52%
Epoch 10: val_accuracy ≈ 50–55%

Test Accuracy: 0.51 (51%)

"""
