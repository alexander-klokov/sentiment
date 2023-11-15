from keras.layers import Embedding,SimpleRNN,Dense
from keras.models import Sequential

# from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import numpy as np

from sentiment_embeddings import get_embeddings
from sentiment_utils import FILE_NAME_MODEL, get_dataset_split, max_len, embedding_dim

# get the dataset
X_train, Y_train = get_dataset_split('train', batch_size=12000)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

X_train_indices = tokenizer.texts_to_sequences(X_train)
X_train_indices = pad_sequences(X_train_indices, maxlen=max_len, padding='post')

# Create an embedding matrix
word_index = tokenizer.word_index
num_words = len(word_index) + 1
embedding_matrix = np.zeros((num_words, embedding_dim))

embedding_index = get_embeddings()

for word, i in word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

embedding = Embedding(
    num_words, 
    embedding_dim, 
    weights=[embedding_matrix],
    input_length=max_len,
    trainable=False
)

# build the model
model = Sequential()
model.add(embedding)
model.add(SimpleRNN(32))
model.add(Dense(28,activation='softmax'))

print(model.summary())

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

# train the model
model.fit(
    X_train_indices, 
    Y_train,
    validation_split=0.2,
    epochs=5
)

model.save(FILE_NAME_MODEL)
