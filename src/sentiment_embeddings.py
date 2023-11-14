import os
import numpy as np

path_to_data = os.environ["DATA_TEXT_GLOVE"] + '/glove.6B.50d.txt'


def get_embeddings():

    embeddings = {}
    with open(path_to_data, encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            coeffs = np.asarray(values[1:], dtype='float32')
            embeddings[word] = coeffs

    return embeddings
