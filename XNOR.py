from gc import callbacks
import tensorboard
import tensorflow as tf
from keras import layers
from keras.layers import Activation, Dense
import numpy as np
from time import time
from tensorflow.python.keras.callbacks import TensorBoard

# Premèire étape : chargez et mettez en forme vos données.

x = np.array(([0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]), dtype=float)
y = np.array(([1], [0], [0], [0], [0], [0], [0], [1]), dtype=float)

# Deuxième étape : Définissez votre modèle de réseau de neurones et ses couches.

model = tf.keras.Sequential()
model.add(Dense(4, input_dim=3, activation="relu", use_bias=True))
model.add(Dense(4, activation="relu", use_bias=True))
model.add(Dense(1, activation='sigmoid', use_bias=True))

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
# Troisième étape : Compilez le modèle.

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy'])

print(model.get_weights())

# Quatrième étape : Faites travailler votre modèle et enraïnez le.

history = model.fit(x, y, epochs=2000, validation_data=(x, y), callbacks = [tensorboard])

model.summary()

loss_history = history.history["loss"]
numpy_loss_history = np.array(loss_history)
np.savetxt("loss_history.txt", numpy_loss_history, delimiter="\n")

binary_accuracy_history = history.history["binary_accuracy"]
numpy_binary_accuracy = np.array(binary_accuracy_history)
np.savetxt("binary_accuracy.txt", numpy_binary_accuracy, delimiter="\n")

# Cinquième étape : Evaluez le modèle.

print(np.mean(history.history["binary_accuracy"]))

z = np.array(([0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1], [0, 1, 1], [1, 0, 0]), dtype=float)
result = model.predict(z).round()

print(result)
