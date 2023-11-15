from keras.models import Sequential
from keras.layers import SimpleRNN, Dense

# from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sentiment_layer_embedding import get_layer_embedding
from sentiment_utils import FILE_NAME_MODEL, get_dataset_split, max_len


# get the training dataset
X_train, Y_train = get_dataset_split('train', batch_size=1200)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

# X_train_indices: [[1, 2, 3, 0, 0, ...], [1, 5, 4, 0, 0], ...] for each sentence in data
X_train_indices = tokenizer.texts_to_sequences(X_train)
X_train_indices = pad_sequences(X_train_indices, maxlen=max_len, padding='post')

# make the embedding layer
embedding = get_layer_embedding(tokenizer)

# build the model
model = Sequential()
model.add(embedding)
model.add(SimpleRNN(32))
model.add(Dense(28, activation='softmax'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

print(model.summary())

# train the model
model.fit(
    X_train_indices, 
    Y_train,
    validation_split=0.2,
    epochs=5
)

# save the model
model.save(FILE_NAME_MODEL)
