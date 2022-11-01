import tensorflow as tf
from tensorflow.keras.layers import Layer


class LayerNormalization(Layer):
    """
    Implement layer-normalization
    """

    def __init__(self, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        super(LayerNormalization, self).build(input_shape)
        self.scale = self.add_weight(name='layer_norm_scale',
                                     shape=(input_shape[-1]),
                                     initializer=tf.ones_initializer(),
                                     trainable=True)
        self.b = self.add_weight(name='layer_norm_bias',
                                 shape=(input_shape[-1]),
                                 initializer=tf.zeros_initializer(),
                                 trainable=True)

    def call(self, inputs, *args, **kwargs):
        # compute the layer's mean and variance
        layer_mean = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        layer_var = tf.reduce_mean(tf.square(inputs - layer_mean), axis=-1, keepdims=True)
        # layer normalization
        layer_norm = (inputs - layer_mean) * tf.math.rsqrt(layer_var + 1e-7)
        layer_norm = layer_norm * self.scale + self.b

        return layer_norm
