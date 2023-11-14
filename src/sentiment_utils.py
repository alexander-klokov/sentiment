import tensorflow as tf
import tensorflow_datasets as tfds

FILE_NAME_MODEL = 'model/model.keras'

emotions = [
    'admiration',
    'amusement',
    'anger',
    'annoyance',
    'approval',
    'caring',
    'confusion',
    'curiosity',
    'desire',
    'disappointment',
    'disapproval',
    'disgust',
    'embarrassment',
    'excitement',
    'fear',
    'gratitude',
    'grief',
    'joy',
    'love',
    'nervousness',
    'optimism',
    'pride',
    'realization',
    'relief',
    'remorse',
    'sadness',
    'surprise',
    'neutral',
]

def one_hot_encode(x):

    vec = tf.stack([x[emotion] for emotion in emotions], 0)

    return x['comment_text'], tf.cast(vec, tf.uint8)


def preprocess_dataset(split, batch_size=128):

    ds = tfds.load('goemotions', split=split)

    ds = ds.map(one_hot_encode, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(buffer_size=batch_size * 10)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return ds