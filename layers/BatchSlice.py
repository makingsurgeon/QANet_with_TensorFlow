import tensorflow as tf
from tensorflow.keras.layers import Layer


class BatchSlice(Layer):
    """
    Slice date from the input of each batch
    Number of inputs is batch_size
    Inputs is a list of [context,context_length] or [question, question_length]
    """

    def __init__(self, dim=2, **kwargs):
        super(BatchSlice, self).__init__(**kwargs)
        self.dim = dim

    def build(self, input_shape):
        super(BatchSlice, self).build(input_shape)

    def call(self, inputs, *args, **kwargs):
        x, length = inputs
        max_len = tf.cast(tf.reduce_max(length), tf.int32)  # get the longest length
        begin = [0] * self.dim
        size = [-1] * self.dim
        size[1] = max_len  # include all elements in the slice
        outputs = tf.slice(x, begin, size)

        return outputs
