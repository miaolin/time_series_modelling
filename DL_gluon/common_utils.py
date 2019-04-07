# coding=utf-8
import numpy as np
from mxnet import nd


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
    return - nd.nansum( y * nd.log_softmax(yhat_linear), axis=0, exclude=True)