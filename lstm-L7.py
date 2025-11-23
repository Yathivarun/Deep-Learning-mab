import tensorflow as tf
from tensorflow.keras import layers, models

# --------------------------------------------------------
# 1. LOAD IMDB DATASET
# --------------------------------------------------------
max_words = 10000     # Only keep top 10k words
max_len = 200         # Pad/cut each review to length 200

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
    num_words=max_words
)

# Convert lists → padded integer sequences
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)


# --------------------------------------------------------
# 2. BUILD LSTM MODEL
# --------------------------------------------------------
model = models.Sequential([
    layers.Embedding(max_words, 128, input_length=max_len),   # word embeddings
    layers.LSTM(128),                                         # LSTM layer
    layers.Dense(1, activation="sigmoid")                     # binary output
])

model.summary()


# --------------------------------------------------------
# 3. COMPILE THE MODEL
# --------------------------------------------------------
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# --------------------------------------------------------
# 4. TRAIN THE MODEL
# --------------------------------------------------------
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2,
    verbose=2
)


# --------------------------------------------------------
# 5. TEST ACCURACY
# --------------------------------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("\nTest Accuracy:", test_acc)


"""
Epoch 1: val_accuracy ~ 0.86
Epoch 2: val_accuracy ~ 0.88
Epoch 3: val_accuracy ~ 0.89
Epoch 4: val_accuracy ~ 0.90
Epoch 5: val_accuracy ~ 0.90

Test Accuracy: ~0.89 to 0.91


"""
