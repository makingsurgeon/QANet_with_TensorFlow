from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow.keras import backend


class ContextQueryAttention(Layer):
    """
    Implement context-to-query attention and query-to-context attention
    First compute the similarities between each pair of context and query word, rendering a
    similar matrix S, then normalized each row of of S by applying the softmax function, getting
    a matrix S_bar, then the context-to-query attention is cimputed as S_bar·Q.T.
    Second, compute the column normalized matrix S_bar_bar of S by softmax function, and the
    query-to-attention is S_bar·S_bar_bar·C.T
    """

    def __init__(self, max_cont_len, max_que_len, output_dim, **kwargs):
        """
        max_cont_len: max context word
        max_que_len: max question word
        """
        super(ContextQueryAttention, self).__init__(**kwargs)
        self.max_cont_len = max_cont_len
        self.max_que_len = max_que_len
        self.output_dim = output_dim

    def build(self, input_shape):
        # input_shape: [(None, max_len , 128), (None, max_len , 128)]
        super(ContextQueryAttention, self).build(input_shape)
        initializer = VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform')
        self.W0 = self.add_weight(name='W0',
                                  shape=(input_shape[0][-1], 1),
                                  initializer=initializer,
                                  regularizer=l2(3e-7),
                                  trainable=True)
        self.W1 = self.add_weight(name='W1',
                                  shape=(input_shape[1][-1], 1),
                                  initializer=initializer,
                                  regularizer=l2(3e-7),
                                  trainable=True)
        self.W2 = self.add_weight(name='W2',
                                  shape=(1, 1, input_shape[0][-1]),
                                  initializer=initializer,
                                  regularizer=l2(3e-7),
                                  trainable=True)
        self.b = self.add_weight(name='bias',
                                 shape=([1]),
                                 initializer=tf.zeros_initializer(),
                                 regularizer=l2(3e-7),
                                 trainable=True)

    def make_mask(self, inputs, mask, mask_value=-1e30):
        mask = tf.cast(mask, tf.float32)
        outputs = inputs + (1 - mask) * mask_value
        return outputs

    def call(self, inputs, *args, **kwargs):
        cont, ques, cont_mask, ques_mask = inputs
        # get similarity matrix S
        # [batch_size, max_cont_word, word_dimension] -> [batch_size, max_cont_word, max_que_word, word_dimension]
        context_expand = tf.tile(backend.dot(cont, self.W0), [1, 1, self.max_que_len])
        # [batch_size, max_que_word, word_dimension] -> [batch_size, max_cont_word, max_que_word, word_dimension]
        query_expand = tf.tile(tf.transpose(backend.dot(ques, self.W1), perm=(0, 2, 1)),
                               [1, self.max_cont_len, 1])
        mat = backend.batch_dot(cont * self.W2, tf.transpose(ques, perm=(0, 2, 1)))
        mat_S = context_expand + query_expand + mat
        mat_S += self.b
        ques_mask = tf.expand_dims(ques_mask, 1)
        # normalizing by applying softmax over rows of similarity matrix
        S_bar = tf.nn.softmax(self.make_mask(mat_S, ques_mask))
        cont_mask = tf.expand_dims(cont_mask, 2)
        # normalizing by applying softmax over columns of similarity matrix
        S_bar_bar = tf.transpose(tf.nn.softmax(self.make_mask(mat_S, cont_mask), axis=1), perm=(0, 2, 1))
        # compute the context to query attention
        cont_to_que = tf.matmul(S_bar, ques)
        # compute the query to context attention
        que_to_cont = tf.matmul(tf.matmul(S_bar, S_bar_bar), cont)
        # computing B = S_bar * S_bar_bar * Context
        mat_B = tf.concat([cont, cont_to_que, cont * cont_to_que, cont * que_to_cont], axis=-1)
        # result = tf.concat([x_cont, c2q, x_cont * c2q, x_cont * q2c], axis=-1)

        return mat_B
