from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.regularizers import l2
import tensorflow as tf


class DepthwiseConv(Layer):
    """
    Implement the depthwise separable convolution
    """

    def __init__(self, filters, kernel_size, **kwargs):
        super(DepthwiseConv, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size

    def build(self, input_shape):
        super(DepthwiseConv, self).build(input_shape)
        initializer = VarianceScaling(scale=2.0, mode='fan_in', distribution='normal')
        self.w_depthwise = self.add_weight(name="depthwise_filter",
                                           shape=(self.kernel_size, 1, input_shape[-1], 1),
                                           initializer=initializer,
                                           regularizer=l2(3e-7),
                                           trainable=True)
        self.w_pointwise = self.add_weight(name="pointwise_filter",
                                           shape=(1, 1, input_shape[-1], self.filters),
                                           initializer=initializer,
                                           regularizer=l2(3e-7),
                                           trainable=True)
        self.b = self.add_weight(name="bias",
                                 shape=input_shape[-1],
                                 regularizer=l2(3e-7),
                                 initializer=tf.zeros_initializer())

    def call(self, inputs, *args, **kwargs):
        x = tf.expand_dims(inputs, axis=2)
        # Performs a depthwise convolution, followed by a pointwise convolution
        x = tf.nn.separable_conv2d(x, self.w_depthwise, self.w_pointwise,
                                   strides=(1, 1, 1, 1), padding='SAME')
        x += self.b
        x = tf.nn.relu(x)
        outputs = tf.squeeze(x, axis=2)

        return outputs
