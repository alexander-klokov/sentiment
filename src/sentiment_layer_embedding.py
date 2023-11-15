import os
import numpy as np

from keras.layers import Embedding

from sentiment_utils import max_len

path_to_data = os.environ["DATA_TEXT_GLOVE"] + '/glove.6B.50d.txt'
embedding_dim = 50

def get_embedding_index():

    embedding_index = {}
    with open(path_to_data, encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            coeffs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coeffs

    return embedding_index


def get_layer_embedding(tokenizer) :

    embedding_index = get_embedding_index() # from GloVe: {'the': [0.1, 0.03, ...], ...}

    word_index = tokenizer.word_index # from data: {'this': 1, 'my': 2, 'dog': 3, ...}
    num_words = len(word_index) + 1
    embedding_matrix = np.zeros((num_words, embedding_dim))

    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # embedding_matrix: word from data - corresponding vector

    return Embedding(
        num_words, 
        embedding_dim, 
        weights=[embedding_matrix],
        input_length=max_len,
        trainable=False
    )