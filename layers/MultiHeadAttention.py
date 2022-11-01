from tensorflow.keras.layers import Layer
import tensorflow as tf


class MultiHeadAttention(Layer):
    """
    Multi-head attention for the self-attention layer.
    For each position in the inputs, called the query, computes a weighted sum of all
    positions, or keys. Then compute the similarity between the query and key.

    """
    def __init__(self, units, num_heads, dropout=0.0, add_bias=True, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.dropout = dropout
        self.add_bias = add_bias

    def build(self, input_shape):
        super(MultiHeadAttention, self).build(input_shape)
        self.b = self.add_weight(name='bias',
                                 shape=([1]),
                                 initializer=tf.zeros_initializer())

    def make_mask(self, inputs, mask, mask_value=-1e30):
        mask = tf.cast(mask, tf.float32)
        outputs = inputs + (1 - mask) * mask_value
        return outputs

    def split_last_dim(self, x, num_heads):
        shape = x.get_shape().dims
        last_dim = shape[-1]
        if last_dim:  # if the last dim is not None
            new_shape = shape[:-1] + [num_heads] + [last_dim // num_heads]
        else:
            new_shape = shape[:-1] + [num_heads]
        split_result = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [num_heads, -1]], 0))
        split_result.set_shape(new_shape)
        outputs = tf.transpose(split_result, [0, 2, 1, 3])
        return outputs

    def combine_dims(self, x):
        # comnine the last two dimensions of inputs
        shape = x.get_shape().dims
        # get the last two dimensions
        dim1, dim2 = shape[-2:]
        if dim1 and dim2:  # if both dim1 and dim2 are not None
            new_shape = shape[:-2] + [dim1 * dim2]
        else:
            new_shape = shape[:-2]
        combine_result = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
        combine_result.set_shape(new_shape)

        return combine_result

    def dot_product(self, inputs, mask=None, dropout=0.0, training=None):
        # compute the similarity between the query and key
        Q, K, V = inputs
        logits = tf.matmul(Q, K, transpose_b=True)  # [batch_size, num_heads, max_len, max_len]
        if self.add_bias:
            logits += self.b
        if mask is not None:
            mask = tf.expand_dims(mask, axis=1)
            mask = tf.expand_dims(mask, axis=1)  # [batch_size, 1, 1, max_len]
            logits = self.make_mask(logits, mask)
        # apply softmax normalization
        w = tf.nn.softmax(logits, name="attention_weights")
        # when training, apply dropout, when test, no dropout
        w = tf.keras.backend.in_train_phase(tf.nn.dropout(w, dropout), w, training=training)
        # apply weights on values
        output = tf.matmul(w, V)

        return output

    def call(self, inputs, mask=None, training=None):
        memory, query, seq_mask = inputs
        Q = self.split_last_dim(query, self.num_heads)
        memory = tf.split(memory, 2, axis=2)
        K = self.split_last_dim(memory[0], self.num_heads)
        V = self.split_last_dim(memory[1], self.num_heads)

        key_depth_per_head = self.units // self.num_heads
        Q *= (key_depth_per_head ** -0.5)
        outputs = self.dot_product([Q, K, V], seq_mask, dropout=self.dropout, training=training)
        outputs = self.combine_dims(tf.transpose(outputs, [0, 2, 1, 3]))

        return outputs
