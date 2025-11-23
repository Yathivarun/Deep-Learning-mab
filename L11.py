"""
dataset/
    train/
        class1/
        class2/
        class3/
        ...
    test/
        class1/
        class2/
        class3/
        ...

"""

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Path to your dataset folder
dataset_path = "/mnt/data/dataset"  # change this to your path

img_size = (128, 128)   # Satellite images can be resized
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path + "/train",
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path + "/test",
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False
)

# Class names
class_names = train_ds.class_names
print("Classes:", class_names)
num_classes = len(class_names)

train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds  = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)


data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

model = models.Sequential([
    data_augmentation,

    layers.Rescaling(1./255),

    layers.Conv2D(32, (3,3), activation="relu", padding="same"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation="relu", padding="same"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation="relu", padding="same"),
    layers.MaxPooling2D(),

    layers.Conv2D(256, (3,3), activation="relu", padding="same"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),                       # prevents overfitting

    layers.Dense(num_classes, activation="softmax")
])

model.summary()


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


history = model.fit(
    train_ds,
    epochs=20,
    validation_data=test_ds,
    verbose=2
)


test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print("Test Accuracy:", test_acc)



plt.figure(figsize=(14,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy Curve")
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss Curve")
plt.legend()

plt.show()




