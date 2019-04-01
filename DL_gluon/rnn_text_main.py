# coding=utf-8
import mxnet as mx
from DL_gluon.text_class import Corpus

# some configurations
args_data = '../data/nlp/ptb.'
args_model = 'rnn_relu'
args_emsize = 100
args_nhid = 100
args_nlayers = 2
args_lr = 1.0
args_clip = 0.2
args_epochs = 1
args_batch_size = 32
args_bptt = 5
args_dropout = 0.2
args_tied = True
args_cuda = 'store_true'
args_log_interval = 500
args_save = 'model.param'


def batchify(data, batch_size):
    """
    reshape data into (number_example, batch_size)
    :param data:
    :param batch_size:
    :return:
    """
    nbatch = data.shape[0] // batch_size
    data = data[:nbatch * batch_size]
    data = data.reshape((batch_size, nbatch)).T
    return data


if __name__ == "__main__":

    context = mx.gpu() # use gpu
    corpus = Corpus(args_data)

    train_data = batchify(corpus.train, args_batch_size).as_in_context(context)
    val_data = batchify(corpus.valid, args_batch_size).as_in_context(context)
    test_data = batchify(corpus.test, args_batch_size).as_in_context(context)





