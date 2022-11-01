# parameters of the model

class Params(object):
    word_dim = 300  # dimension of glove word vector
    char_dim = 200  # dimension of character
    cont_limit = 400  # context word limit
    ques_limit = 64  # question word limit
    char_limit = 16  # char limit in one word
    ans_limit = 30  # answer word limit
    filters = 128  # number of filters
    num_head = 8  # number of heads for attention
    dropout = 0.1  # dropout rate
    batch_size = 12  # batch size
    epoch = 14  # epochs
    learning_rate = 0.001  # learning_rate
    warm_up_steps = 1000  # the warm_up steps of learning rate
