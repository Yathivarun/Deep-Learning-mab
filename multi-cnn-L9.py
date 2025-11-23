import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# Load CIFAR-10 (32x32 color images)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values (important for CNNs)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

y_train = y_train.flatten()
y_test = y_test.flatten()

print("Training samples:", x_train.shape)
print("Testing samples:", x_test.shape)

model = models.Sequential([

    layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),

    layers.Dense(10, activation='softmax')   # 10 classes
])

model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    verbose=2
)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("\nTest Accuracy:", test_acc)


"""
Epoch 1: val_acc ≈ 0.65–0.70
Epoch 5: val_acc ≈ 0.75–0.78
Epoch 10: val_acc ≈ 0.78–0.82

Test Accuracy: ~0.78 to 0.82

"""


from tensorflow.keras.preprocessing import image

img_path = "/mnt/data/new_image.png"  # change to your file

# Load image as 32x32
img = image.load_img(img_path, target_size=(32, 32))

# Convert to array
img_array = image.img_to_array(img) / 255.0

# Add batch dimension: (1, 32, 32, 3)
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)
pred_class = np.argmax(pred)

print("Predicted Class Index:", pred_class)

labels = ["airplane", "automobile", "bird", "cat", "deer",
          "dog", "frog", "horse", "ship", "truck"]

print("Predicted Label:", labels[pred_class])


