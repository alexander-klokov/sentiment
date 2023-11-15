import tensorflow as tf

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sentiment_utils import FILE_NAME_MODEL, get_dataset_split, emotions, max_len

X_test, Y_test = get_dataset_split('train', batch_size=2000)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_test)

X_test_indices = tokenizer.texts_to_sequences(X_test)
X_test_indices = pad_sequences(X_test_indices, maxlen=max_len, padding='post')

# load the model
model = tf.keras.models.load_model(FILE_NAME_MODEL)

#evaluate the model
model.evaluate(X_test_indices, Y_test)

##

predictions = model.predict(X_test_indices)
for prediction in predictions:
    max_index = tf.argmax(prediction)
    print(X_test_indices[max_index], ':', emotions[max_index])
