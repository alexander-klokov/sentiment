import tensorflow as tf
import tensorflow_datasets as tfds

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sentiment_utils import emotions, FILE_NAME_MODEL

phrases = [
    'this is a random phrase',
    'So happy!'
]

max_len=200

tokenizer = tf.keras.preprocessing.text.Tokenizer()

X_test_indices = tokenizer.texts_to_sequences(phrases)
X_test_indices = pad_sequences(X_test_indices, maxlen=max_len, padding='post')

model = tf.keras.models.load_model(FILE_NAME_MODEL)
preds = model.predict(X_test_indices)

for j in range(len(phrases)):
    max_index = tf.argmax(preds[j])
    print(phrases[j], ':', emotions[max_index])