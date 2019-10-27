import tensorflow as tf
from typing import List


def batch_index(values, indices):
    one_hot_indices = tf.one_hot(
        indices, tf.shape(values)[-1], dtype=values.dtype)
    return tf.reduce_sum(values * one_hot_indices, axis=-1)


def any_in_str(candidate_string_list: List[str], target_sting: str):
    return any(x in target_sting for x in candidate_string_list)
