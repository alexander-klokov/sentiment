import sys

import tensorflow as tf

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sentiment_utils import emotions, FILE_NAME_MODEL, max_len

for arg in sys.argv:
    print(arg)

phrase = sys.argv[1]

tokenizer = Tokenizer()

X_test_indices = tokenizer.texts_to_sequences([phrase])
X_test_indices = pad_sequences(X_test_indices, maxlen=max_len, padding='post')

model = tf.keras.models.load_model(FILE_NAME_MODEL)
preds = model.predict(X_test_indices)

max_index = tf.argmax(preds[0])
print(phrase, ':', emotions[max_index])
