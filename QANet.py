import tensorflow as tf
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Embedding, Concatenate, Conv1D, GlobalMaxPooling1D
from layers.BatchSlice import BatchSlice
from layers.ContextQueryAttention import ContextQueryAttention
from layers.DepthwiseConv import DepthwiseConv
from layers.LayerDropout import LayerDropout
from layers.LayerNormalization import LayerNormalization
from layers.MultiHeadAttention import MultiHeadAttention
from layers.OutputLayer import OutputLayer
from layers.PositionEncoding import PositionEncoding
from layers.ZeroPadding import ZeroPadding
from params import Params

tf.config.run_functions_eagerly(True)


def make_mask(inputs, mask, mask_value=-1e30):
    mask = tf.cast(mask, tf.float32)
    output = inputs + (1 - mask) * mask_value
    return output


def highway(highway_layers, inputs, num_layers=2, dropout=0.0):
    # reduce dim
    x = highway_layers[0](inputs)
    for i in range(num_layers):
        T = highway_layers[i * 2 + 1](x)
        H = highway_layers[i * 2 + 2](x)
        H = Dropout(dropout)(H)
        outputs = H * T + x * (1 - T)
    return outputs


def conv(conv_layer, inputs, num_conv=4, dropout=0.0, l=1.0, L=1.0):
    # the convolution layer in the encoder block
    for i in range(num_conv):
        residual = inputs
        # first layer normalization, then conv, then through a residual network
        x = LayerNormalization()(inputs)
        if i % 2 == 0:
            x = Dropout(dropout)(x)
        x = conv_layer[i](x)
        outputs = LayerDropout(dropout * (l / L))([x, residual])
    return outputs


def self_attention(attention_layer, input, mask, dropout=0.0, l=1.0, L=1.0):
    # the self-attention layer in the encoder block
    residual = input
    # first layer normalization, then self-attention, then through a residual network
    x = LayerNormalization()(input)
    x = Dropout(dropout)(x)
    x1 = attention_layer[0](x)
    x2 = attention_layer[1](x)
    x = attention_layer[2]([x1, x2, mask])
    output = LayerDropout(dropout * (l / L))([x, residual])
    return output


def feed_forward(feed_forward_layers, input, dropout=0.0, l=1.0, L=1.0):
    # the feed-ford layer in the encoder layer
    residual = input
    x = LayerNormalization()(input)
    x = Dropout(dropout)(x)
    x = feed_forward_layers[0](x)
    x = feed_forward_layers[1](x)
    output = LayerDropout(dropout * (l / L))([x, residual])
    return output


params = Params()


class QANet(Model):
    def __init__(self, params, word_mat=None, char_mat=None, **kwargs):
        # word_mat: word embedding matrix
        # char_mat: character embedding matrix
        super(QANet, self).__init__(**kwargs)
        self.word_dim = params.word_dim
        self.char_dim = params.char_dim
        self.char_dim = params.char_dim
        self.cont_limit = params.cont_limit
        self.char_limit = params.char_limit
        self.ans_limit = params.ans_limit
        self.filters = params.filters
        self.num_head = params.num_head
        self.dropout = params.dropout
        self.word_mat = word_mat
        self.char_mat = char_mat

        self.regularizer = l2(3e-7)
        self.init = VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform')
        self.init_relu = VarianceScaling(scale=2.0, mode='fan_in', distribution='normal')

        # Initialize layers
        self.batch_slice2 = BatchSlice(dim=2)
        self.batch_slice3 = BatchSlice(dim=3)
        self.qa_output = OutputLayer(self.ans_limit, name='qa_output')
        self.depwise_conv7 = DepthwiseConv(self.filters, 7)
        self.depwise_conv5 = DepthwiseConv(self.filters, 5)
        self.start_label_pad = ZeroPadding(self.cont_limit, name='start_pos')
        self.end_label_pad = ZeroPadding(self.cont_limit, name='end_pos')
        self.word_embed = Embedding(self.word_mat.shape[0], self.word_dim,
                                    weights=[self.word_mat], trainable=False,
                                    name='word_embedding')
        self.concat = Concatenate()
        self.char_embed = Embedding(self.char_mat.shape[0], self.char_dim,
                                    weights=[self.char_mat], name='char_embedding')
        self.char_conv1 = Conv1D(self.filters, 5,
                                 activation='relu',
                                 kernel_initializer=self.init_relu,
                                 kernel_regularizer=self.regularizer,
                                 name='char_conv')
        self.char_conv2 = Conv1D(self.filters, 5,
                                 activation='relu',
                                 kernel_initializer=self.init_relu,
                                 kernel_regularizer=self.regularizer,
                                 name='char_conv')
        self.glo_max_pool = GlobalMaxPooling1D()
        self.highway_conv1 = Conv1D(self.filters, 1,
                                    kernel_initializer=self.init,
                                    kernel_regularizer=self.regularizer,
                                    name='highway_input_projection')
        self.highway_conv2 = Conv1D(self.filters, 1,
                                    kernel_initializer=self.init,
                                    kernel_regularizer=self.regularizer,
                                    activation='sigmoid',
                                    name='highway0_gate')
        self.highway_conv3 = Conv1D(self.filters, 1,
                                    kernel_initializer=self.init,
                                    kernel_regularizer=self.regularizer,
                                    activation='linear',
                                    name='highway0_linear')
        self.highway_conv4 = Conv1D(self.filters, 1,
                                    kernel_initializer=self.init,
                                    kernel_regularizer=self.regularizer,
                                    activation='sigmoid',
                                    name='highway1_gate')
        self.highway_conv5 = Conv1D(self.filters, 1,
                                    kernel_initializer=self.init,
                                    kernel_regularizer=self.regularizer,
                                    activation='linear',
                                    name='highway1_linear')
        self.self_atten_conv1 = Conv1D(2 * self.filters, 1,
                                       kernel_initializer=self.init,
                                       kernel_regularizer=self.regularizer)
        self.self_atten_conv2 = Conv1D(self.filters, 1,
                                       kernel_initializer=self.init,
                                       kernel_regularizer=self.regularizer)
        self.multi_head_atten1 = MultiHeadAttention(self.filters, self.num_head,
                                                    dropout=self.dropout, add_bias=False)
        self.feedforward_conv1 = Conv1D(self.filters, 1,
                                        kernel_initializer=self.init,
                                        kernel_regularizer=self.regularizer,
                                        activation='relu')
        self.feedforward_conv2 = Conv1D(self.filters, 1,
                                        kernel_initializer=self.init,
                                        kernel_regularizer=self.regularizer,
                                        activation='linear')
        self.feedforward_conv3 = Conv1D(self.filters, 1,
                                        kernel_initializer=self.init,
                                        kernel_regularizer=self.regularizer,
                                        activation='relu')
        self.feedforward_conv4 = Conv1D(self.filters, 1,
                                        kernel_initializer=self.init,
                                        kernel_regularizer=self.regularizer,
                                        activation='linear')
        self.cont2que_atten_conv = Conv1D(self.filters, 1,
                                          kernel_initializer=self.init,
                                          kernel_regularizer=self.regularizer,
                                          activation='linear')
        self.pos_encode = PositionEncoding()
        self.self_atten_conv3 = Conv1D(2 * self.filters, 1,
                                       kernel_initializer=self.init,
                                       kernel_regularizer=self.regularizer)
        self.self_atten_conv4 = Conv1D(self.filters, 1,
                                       kernel_initializer=self.init,
                                       kernel_regularizer=self.regularizer)
        self.multi_head_atten2 = MultiHeadAttention(self.filters, self.num_head,
                                                    dropout=self.dropout, add_bias=False)
        self.out_start_conv = Conv1D(1, 1,
                                     kernel_initializer=self.init,
                                     kernel_regularizer=self.regularizer,
                                     activation='linear')
        self.out_end_conv = Conv1D(1, 1,
                                   kernel_initializer=self.init,
                                   kernel_regularizer=self.regularizer,
                                   activation='linear')

    def call(self, inputs, training=True, mask=None):
        cont_word, ques_word, cont_char, ques_char = inputs

        # get mask
        cont_mask = tf.cast(cont_word, tf.bool)
        ques_mask = tf.cast(ques_word, tf.bool)
        # get length of context and query
        cont_len = tf.expand_dims(tf.reduce_sum(tf.cast(cont_mask, tf.int32), axis=1), axis=1)
        ques_len = tf.expand_dims(tf.reduce_sum(tf.cast(ques_mask, tf.int32), axis=1), axis=1)
        # slice
        cont_word_input = self.batch_slice2([cont_word, cont_len])
        ques_word_input = self.batch_slice2([ques_word, ques_len])
        cont_char_input = self.batch_slice3([cont_char, cont_len])
        ques_char_input = self.batch_slice3([ques_char, ques_len])
        # get mask after slice
        cont_mask = self.batch_slice2([cont_mask, cont_len])
        ques_mask = self.batch_slice2([ques_mask, ques_len])
        # get length of context and query after slice
        cont_max_len = tf.cast(tf.reduce_max(cont_len), tf.int32)
        ques_max_len = tf.cast(tf.reduce_max(ques_len), tf.int32)
        # embedding word
        embed_cont_word = self.word_embed(cont_word_input)
        embed_ques_word = self.word_embed(ques_word_input)
        # embedding character
        embed_cont_char = self.char_embed(cont_char_input)
        embed_ques_char = self.char_embed(ques_char_input)
        embed_cont_char = tf.reshape(embed_cont_char, (-1, self.char_limit, self.char_dim))
        embed_ques_char = tf.reshape(embed_ques_char, (-1, self.char_limit, self.char_dim))
        embed_cont_char = self.char_conv1(embed_cont_char)
        embed_ques_char = self.char_conv2(embed_ques_char)
        embed_cont_char = self.glo_max_pool(embed_cont_char)
        embed_ques_char = self.glo_max_pool(embed_ques_char)
        embed_cont_char = tf.reshape(embed_cont_char, (-1, cont_max_len, self.filters))
        embed_ques_char = tf.reshape(embed_ques_char, (-1, ques_max_len, self.filters))

        # highwayNet
        cont_input = self.concat([embed_cont_word, embed_cont_char])
        ques_input = self.concat([embed_ques_word, embed_ques_char])

        # highway shared layers
        highway_layers = [self.highway_conv1, self.highway_conv2, self.highway_conv3,
                          self.highway_conv4, self.highway_conv5]

        cont_input = highway(highway_layers, cont_input, num_layers=2, dropout=self.dropout)
        ques_input = highway(highway_layers, ques_input, num_layers=2, dropout=self.dropout)

        # build shared layers
        # shared convolution
        encoder_conv1 = []
        for i in range(4):
            encoder_conv1.append(self.depwise_conv7)
        # shared attention
        encoder_self_attention1 = [self.self_atten_conv1, self.self_atten_conv2, self.multi_head_atten1]
        # shared feed-forward
        encoder_feed_forward1 = [self.feedforward_conv1, self.feedforward_conv2]

        # Context Embedding Encoder Layer
        cont_input = self.pos_encode(cont_input)
        cont_input = conv(encoder_conv1, cont_input, 4, self.dropout)
        cont_input = self_attention(encoder_self_attention1, cont_input, cont_mask, self.dropout)
        cont_input = feed_forward(encoder_feed_forward1, cont_input, self.dropout)

        # Question Embedding Encoder Layer
        ques_input = self.pos_encode(ques_input)
        ques_input = conv(encoder_conv1, ques_input, 4, self.dropout)
        ques_input = self_attention(encoder_self_attention1, ques_input, ques_mask, self.dropout)
        ques_input = feed_forward(encoder_feed_forward1, ques_input, self.dropout)

        # Context_to_Query_Attention_Layer
        # x = context2query_attention(512, c_maxlen, q_maxlen, self.dropout)([x_cont, x_ques, c_mask, q_mask])
        self.cont2que_atten = ContextQueryAttention(cont_max_len, ques_max_len, 512)
        x = self.cont2que_atten([cont_input, ques_input, cont_mask, ques_mask])
        x = self.cont2que_atten_conv(x)

        # Model_Encoder_Layer
        # shared layers
        encoder_conv2 = []
        encoder_self_attention2 = []
        encoder_feed_forward2 = []
        for i in range(7):
            conv_list = []
            for i in range(2):
                conv_list.append(self.depwise_conv5)
            encoder_conv2.append(conv_list)
            encoder_self_attention2.append([self.self_atten_conv3, self.self_atten_conv4, self.multi_head_atten2])
            encoder_feed_forward2.append([self.feedforward_conv3, self.feedforward_conv4])

        outputs = [x]
        for i in range(3):
            x = outputs[-1]
            for j in range(7):
                x = self.pos_encode(x)
                x = conv(encoder_conv2[j], x, 2, self.dropout, l=j, L=7)
                x = self_attention(encoder_self_attention2[j], x, cont_mask, self.dropout, l=j, L=7)
                x = feed_forward(encoder_feed_forward2[j], x, self.dropout, l=j, L=7)
            outputs.append(x)

        # Output_Layer
        x_start = self.concat([outputs[1], outputs[2]])
        x_start = self.out_start_conv(x_start)
        x_start = tf.squeeze(x_start, axis=-1)
        x_start = make_mask(x_start, cont_mask)
        x_start = tf.nn.softmax(x_start)

        x_end = self.concat([outputs[1], outputs[3]])
        x_end = self.out_end_conv(x_end)
        x_end = tf.squeeze(x_end, axis=-1)
        x_end = make_mask(x_end, cont_mask)
        x_end = tf.nn.softmax(x_end)

        x_start_fin, x_end_fin = self.qa_output([x_start, x_end])

        # if use model.fit, the output shape must be padded to the max length
        x_start = self.start_label_pad(x_start)
        x_end = self.end_label_pad(x_end)

        return [x_start, x_end, x_start_fin, x_end_fin]
