import tensorflow as tf

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sentiment_utils import FILE_NAME_MODEL, get_dataset_split

sentences_test, Y_test = get_dataset_split('test', batch_size=1024)
test_data = list(map(lambda s: s.decode("utf-8"), sentences_test))

max_len=200

tokenizer = Tokenizer()

X_test_indices = tokenizer.texts_to_sequences(test_data)
X_test_indices = pad_sequences(X_test_indices, maxlen=max_len, padding='post')

model = tf.keras.models.load_model(FILE_NAME_MODEL)
predictions = model.predict(X_test_indices)

model.evaluate(X_test_indices, Y_test)
