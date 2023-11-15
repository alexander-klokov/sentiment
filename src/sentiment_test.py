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

# visualize test results
X_test_small, Y_test_small = get_dataset_split('train', batch_size=11)

X_test_small_indices = tokenizer.texts_to_sequences(X_test)
X_test_small_indices = pad_sequences(X_test_indices, maxlen=max_len, padding='post')

predictions = model.predict(X_test_small_indices)
for prediction, sentence, labels in zip(predictions, X_test_small, Y_test_small):
    max_index_true = tf.argmax(labels)
    max_index_prediction = tf.argmax(prediction)
    print(sentence, ':', emotions[max_index_prediction], "(", emotions[max_index_true], ")")
