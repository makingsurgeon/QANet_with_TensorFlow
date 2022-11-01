import tensorflow as tf
from tensorflow.keras.layers import Layer


class LayerDropout(Layer):
    """
    The dropout between layers
    """

    def __init__(self, dropout=0.0, **kwargs):
        # if dropout=0., 0% of inputs would be dropped
        super(LayerDropout, self).__init__(**kwargs)
        self.dropout = dropout

    def build(self, input_shape):
        super(LayerDropout, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        x, residual = inputs
        pred = tf.random.uniform([]) < self.dropout
        # if pred is True, there is no dropout
        fn1 = lambda: residual
        # if pred is False, there is dropout
        fn2 = lambda: tf.nn.dropout(x, self.dropout) + residual
        x_train = tf.cond(pred, fn1, fn2)
        x_test = x + residual
        # when training, return x_trian, when testing, return x_test
        outputs = tf.keras.backend.in_train_phase(x_train, x_test, training=training)

        return outputs
