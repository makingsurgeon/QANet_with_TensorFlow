from tensorflow.python.keras.layers import Layer
import tensorflow as tf


class ZeroPadding(Layer):
    """
    Zero padding the input context or query to keep them the same length
    """

    def __init__(self, max_len, **kwargs):
        super(ZeroPadding, self).__init__(**kwargs)
        self.max_len = max_len

    def build(self, input_shape):
        super(ZeroPadding, self).build(input_shape)

    def get_shape_list(self, inputs):
        # Return list of dims
        x = tf.convert_to_tensor(inputs)
        # if the shape is dynamic
        if x.get_shape().dims is None:
            return tf.shape(x)
        # return where the dim is static
        static_shape = x.get_shape().as_list()
        shape = tf.shape(x)
        shape_list = []
        for i, dim in enumerate(static_shape):
            if dim is None:
                shape_list.append(shape[i])
            else:
                shape_list.append(dim)
        return shape_list

    def call(self, inputs, *args, **kwargs):
        # zero padding the context or query, so they have the same length respectively
        list_shape = self.get_shape_list(inputs)
        # padding to the max length
        zero_paddings = tf.zeros((list_shape[0], self.max_len - list_shape[1]))
        outputs = tf.concat([inputs, zero_paddings], axis=-1)

        return outputs
