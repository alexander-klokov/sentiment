import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sentiment_layer_embedding import get_layer_embedding
from sentiment_utils import FILE_NAME_MODEL, emotions, get_dataset_split, max_len

# get the training dataset
X_train, Y_train = get_dataset_split('train', batch_size=32000)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

# X_train_indices: [[1, 2, 3, 0, 0, ...], [1, 5, 4, 0, 0], ...] for each sentence in data
X_train_indices = tokenizer.texts_to_sequences(X_train)
X_train_indices = pad_sequences(X_train_indices, maxlen=max_len, padding='post')

# build the model
layer_embedding = get_layer_embedding(tokenizer)

model = Sequential()
model.add(layer_embedding)
model.add(LSTM(128,return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(128))
model.add(Dropout(0.3))
model.add(Dense(len(emotions), activation='softmax'))

optimizer = keras.optimizers.Adam(learning_rate=1E-4)
loss = keras.losses.CategoricalCrossentropy(from_logits=False)
metrics = [
    keras.metrics.CategoricalAccuracy('accuracy', dtype=tf.float32)
]

model.compile(optimizer, loss, metrics=metrics)

print(model.summary())

# train the model
callbacks = [
    ModelCheckpoint(FILE_NAME_MODEL, monitor='loss', verbose=0, save_best_only=True, save_weights_only=False),
    EarlyStopping(monitor='accuracy', patience=1)
]

hist = model.fit(
    X_train_indices, 
    Y_train,
    validation_split=0.1,
    epochs=11,
    callbacks=callbacks
)

# save the model
model.save(FILE_NAME_MODEL)
