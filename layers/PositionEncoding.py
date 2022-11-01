import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np


class PositionEncoding(Layer):
    """
    Position Encoding
    The input X: shape [n, d],  with d-dimensional embeddings for n tokens of a sequence.
    The positional encoding matrix P: shape [n, d], with element p(i, 2j) = sin(i/10000^(2j/d)),
                                      p(i, 2j+1) = cos(i/10000^(2j/d)).
    The output of positional encoding is the summation of P and X: X + P.
    """

    def __init__(self, denominator=1.0e4, **kwargs):
        super(PositionEncoding, self).__init__(**kwargs)
        self.denominator = denominator  # the denominator of X in sin(X)

    def build(self, input_shape):
        # [batch_size, seq_length, dimension]
        super(PositionEncoding, self).build(input_shape)

    def call(self, inputs, *args, **kwargs):
        length = tf.shape(inputs)[1]
        dims = tf.shape(inputs)[2]
        # the position encoding matrix
        P = np.zeros((1, length, dims))
        X = np.arange(length).reshape(-1, 1) / np.power(self.denominator, np.arange(0, dims, 2) / dims)
        P[:, :, 0::2] = np.sin(X)
        P[:, :, 1::2] = np.cos(X)

        outputs = inputs + P

        return outputs
