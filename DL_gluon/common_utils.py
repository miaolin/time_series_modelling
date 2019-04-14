# coding=utf-8
import mxnet as mx
import numpy as np
from mxnet import nd, ndarray


def SGD(model_params, learning_rate):
    for param in model_params:
        param[:] = param - learning_rate * param.grad
    return model_params


def transform(data, label):
    return data.astype(np.float32) / 255, label.astype(np.float32)


def relu(X):
    return nd.maximum(X, nd.zeros_like(X))


def softmax(y_linear):
    exp = nd.exp(y_linear - nd.max(y_linear))
    partition = nd.nansum(exp, axis=0, exclude=True).reshape((-1, 1))
    return exp / partition


def cross_entropy(yhat, y):
    return - nd.nansum( y * nd.log(yhat), axis=0, exclude=True)


def softmax_cross_entropy(yhat_linear, y):
    return - nd.nansum(y * nd.log_softmax(yhat_linear), axis=0, exclude=True)


def dropout(X, drop_prob):
    keep_prob = 1.0 - drop_prob
    mask = nd.random_uniform(0, 1.0, X.shape, ctx=X.context) < keep_prob

    if keep_prob > 0.0:
        scale = (1 / keep_prob)
    else:
        scale = 0.0
    return mask * X * scale


def evaluate_accuracy(data_iterator, net, ctx):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx).reshape((-1, 784))
        label = label.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]