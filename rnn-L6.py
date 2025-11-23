import tensorflow as tf
from tensorflow.keras import layers, models

# --------------------------------------------------------
# 1. LOAD DATASET: IMDB Movie Reviews (already tokenized)
# --------------------------------------------------------
max_words = 10000     # use top 10k words
max_len = 200         # cut/pad each review to 200 words

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
    num_words=max_words
)

# Pad sequences so all inputs are equal length
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)


# --------------------------------------------------------
# 2. BUILD SIMPLE RNN MODEL
# --------------------------------------------------------
model = models.Sequential([
    layers.Embedding(max_words, 64, input_length=max_len),  # converts integers → word vectors
    layers.SimpleRNN(64, return_sequences=False),           # RNN layer
    layers.Dense(1, activation="sigmoid")                   # output for binary classification
])

model.summary()


# --------------------------------------------------------
# 3. COMPILE MODEL
# --------------------------------------------------------
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# --------------------------------------------------------
# 4. TRAIN MODEL
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
Epoch 1: val_accuracy ~ 0.81
Epoch 2: val_accuracy ~ 0.84
Epoch 3: val_accuracy ~ 0.85
Epoch 4: val_accuracy ~ 0.86
Epoch 5: val_accuracy ~ 0.87

Test Accuracy: ~0.86

"""
