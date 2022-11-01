import tensorflow as tf
from tensorflow.keras.layers import Layer


class OutputLayer(Layer):
    """
    Predict the probability of each position in the context being the start or end of an
    answer span
    """
    def __init__(self, max_ans_len=30, **kwargs):
        super(OutputLayer, self).__init__(**kwargs)
        self.max_ans_len = max_ans_len

    def build(self, input_shape):
        super(OutputLayer, self).build(input_shape)

    def call(self, inputs, *args, **kwargs):
        start_prob, end_prob = inputs
        # the probability matrix of the answer position
        prob_mat = tf.matmul(tf.expand_dims(start_prob, axis=2), tf.expand_dims(end_prob, axis=1))
        prob_mat = tf.linalg.band_part(prob_mat, 0, self.max_ans_len)
        # find the start and end index with the max probability
        start = tf.argmax(tf.reduce_max(prob_mat, axis=2), axis=1)
        start = tf.cast(start, tf.float32)
        start = tf.reshape(start, (-1, 1))
        end = tf.argmax(tf.reduce_max(prob_mat, axis=1), axis=1)
        end = tf.cast(end, tf.float32)
        end = tf.reshape(end, (-1, 1))

        return [start, end]
