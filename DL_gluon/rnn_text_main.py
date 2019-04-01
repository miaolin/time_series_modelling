# coding=utf-8
import time
import math
import mxnet as mx
from mxnet import gluon, autograd

from DL_gluon.text_class import Corpus
from DL_gluon.rnn_text_model import RNNModel

# some configurations
args_data = '../mxnet-the-straight-dope/data/nlp/ptb.'
args_model = 'rnn_relu'
args_emsize = 100
args_nhid = 100
args_nlayers = 2
args_lr = 1.0
args_clip = 0.2
args_epochs = 10
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


def get_batch(source, i):
    seq_len = min(args_bptt, source.shape[0] - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len]
    return data, target.reshape((-1, ))


def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [i.detach() for i in hidden]
    else:
        hidden = hidden.detach()
    return hidden


def eval(data_source, model):
    total_L = 0.0
    ntotal = 0
    hidden = model.begin_state(func=mx.nd.zeros, batch_size=args_batch_size, ctx=context)

    for i in range(0, data_source.shape[0], args_bptt):
        data, target = get_batch(data_source, i)
        output, hidden = model(data, hidden)
        L = loss(output, target)
        total_L += mx.nd.sum(L).asscalar()
        ntotal += L.size
    return total_L / ntotal


def train(model, train_data, val_data, test_data):

    args_lr = 1.0
    trainer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': args_lr, 'momentum': 0, 'wd': 0})

    best_val = float("Inf")
    for epoch in range(args_epochs):
        total_L = 0.0
        start_time = time.time()
        hidden = model.begin_state(func=mx.nd.zeros, batch_size=args_batch_size, ctx=context)

        for ibatch, i in enumerate(range(0, train_data.shape[0] - 1, args_bptt)):
            data, target = get_batch(train_data, i)
            hidden = detach(hidden)
            with autograd.record():
                output, hidden = model(data, hidden)
                L = loss(output, target)
                L.backward()

            grads = [i.grad(context) for i in model.collect_params().values()]

            # here gradient is for the whole batch
            # so we multiply max_norm by batch_size and bptt size to balance it
            gluon.utils.clip_global_norm(grads, args_clip * args_bptt * args_batch_size)

            trainer.step(args_batch_size)
            total_L += mx.nd.sum(L).asscalar()

            if ibatch % args_log_interval == 0 and ibatch > 0:
                cur_L = total_L / args_bptt / args_batch_size / args_log_interval
                print('[Epoch %d Batch %d] loss %.2f, perplexity %.2f' % (epoch + 1, ibatch, cur_L, math.exp(cur_L)))

                total_L = 0.0

        val_L = eval(val_data, model)
        print('[Epoch %d] time cost %.2fs, validation loss %.2f, validation perplexity '
              '%.2f' % (epoch + 1, time.time() - start_time, val_L, math.exp(val_L)))

        if val_L < best_val:
            best_val = val_L
            test_L = eval(test_data, rnn_model)
            model.save_parameters(args_save)
            print('test loss %.2f, test perplexity %.2f' % (test_L, math.exp(test_L)))
        else:
            args_lr = args_lr * 0.25
            trainer._init_optimizer('sgd', {'learning_rate': args_lr, 'wd': 0})
            model.load_parameters(args_save, context)


if __name__ == "__main__":

    # use gpu
    # context = mx.gpu()
    context = mx.cpu()
    corpus = Corpus(args_data)

    train_data_text = batchify(corpus.train, args_batch_size).as_in_context(context)
    val_data_text = batchify(corpus.valid, args_batch_size).as_in_context(context)
    test_data_text = batchify(corpus.test, args_batch_size).as_in_context(context)

    ntokens = len(corpus.dictionary)
    rnn_model = RNNModel(args_model, vocab_size=ntokens, num_embed=args_emsize, num_hidden=args_nhid,
                         num_layers=args_nlayers, dropout=args_dropout, tie_weights=args_tied)
    rnn_model.collect_params().initialize(mx.init.Xavier(), ctx=context)
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    train(model=rnn_model, train_data=train_data_text, val_data=val_data_text, test_data=test_data_text)
    test_L = eval(test_data_text, rnn_model)
    print("Best test loss %.2f, test perplexity %.2f" % (test_L, math.exp(test_L)))






