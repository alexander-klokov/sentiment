import tensorflow as tf
import tensorflow_datasets as tfds

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import numpy as np

from sentiment_embeddings import get_embeddings
from sentiment_utils import emotions, FILE_NAME_MODEL, preprocess_dataset

max_len=200
embedding_dim = 50

ds_splits = ['train', 'test', 'validation']
datasets = {split: preprocess_dataset(split) for split in ds_splits}

sentences, y_true_batch = next(datasets['train'].as_numpy_iterator())

training_data = sentences
training_data = [
"Interesting I\xe2\x80\x99ve never met any one IRL that met that way. Guess there must be some people meeting that way \xf0\x9f\x98\x80",
"I\xe2\x80\x99d kind of be kind of shocked if Chili wants us to hit mostly ground balls, especially with how hard the shifts are nowadays. ",
"Is he dead now from a tragic drowning accident? Asking for a friend.",
"Ayee it made it ! Enjoy",
"My face imitated the Pikachu meme for a long time...",
"Omg sorry I just laughed out loud and also died inside for you.",
"Oh look me",
"During [NAME] and [NAME] cross examination, [NAME] never contested the finding of the key. Never accused them of planting anything. ",
"I don\xe2\x80\x99t think there is mouse and keyboard support yet",
"Not under [NAME]. Under [NAME] desperate greedy extreme partisan coward politics. They purposely brought racism front and center."
]


tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(training_data)

# Create an embedding matrix
word_index = tokenizer.word_index
num_words = len(word_index) + 1
embedding_matrix = np.zeros((num_words, embedding_dim))

# Create an embedding matrix
word_index = tokenizer.word_index
num_words = len(word_index) + 1
embedding_matrix = np.zeros((num_words, embedding_dim))

embedding_index = get_embeddings()

for word, i in word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

print(word_index)
print(embedding_matrix)

from tensorflow.keras.layers import Embedding,SimpleRNN,Dense
from tensorflow.keras.models import Sequential



print('>>', num_words, embedding_dim)

from tensorflow.keras.callbacks import LambdaCallback

def print_layer_input(epoch, logs):
    input_data = model.layers[1].input_dim
    print(f"Input data at epoch {epoch + 1}:\n{input_data}")

print_input_callback = LambdaCallback(on_epoch_begin=print_layer_input)

model = Sequential()
model.add(Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length=max_len, trainable=False))
model.add(SimpleRNN(32))
model.add(Dense(28,activation='softmax'))
print(model.summary())

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

X_train_indices = tokenizer.texts_to_sequences(training_data)
X_train_indices = pad_sequences(X_train_indices, maxlen=max_len, padding='post')

print('>>>>>', training_data)

Y_train = y_true_batch[:10]

model.fit(X_train_indices, Y_train, batch_size=64, epochs=5)
model.save(FILE_NAME_MODEL)

preds = model.predict(X_train_indices)

print('preds', preds)

for p in preds:
    max_index = tf.argmax(p)
    print(max_index, emotions[max_index])

X_test_indices = tokenizer.texts_to_sequences(['this is the fortch'])
X_test_indices = pad_sequences(X_test_indices, maxlen=max_len, padding='post')

pak = model.predict(X_test_indices)

print(pak)

for p in pak:
    max_index = tf.argmax(p)
    print(max_index, emotions[max_index])