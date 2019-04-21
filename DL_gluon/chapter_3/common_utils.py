# coding=utf-8
from mxnet import nd

from DL_gluon.common_utils import relu, dropout


def net(X, model_params):
    h1_linear = nd.dot(X, model_params[0]) + model_params[1]
    h1 = relu(h1_linear)

    h2_linear = nd.dot(h1, model_params[2]) + model_params[3]
    h2 = relu(h2_linear)

    # omit the softmax function here
    yhat_linear = nd.dot(h2, model_params[4]) + model_params[5]
    return yhat_linear


def net_dropout(X, model_params, drop_prob=0.0):
    h1_linear = nd.dot(X, model_params[0]) + model_params[1]
    h1 = relu(h1_linear)
    h1 = dropout(h1, drop_prob)

    h2_linear = nd.dot(h1, model_params[2]) + model_params[3]
    h2 = relu(h2_linear)
    h2 = dropout(h2, drop_prob)

    # omit the softmax function here
    yhat_linear = nd.dot(h2, model_params[4]) + model_params[5]
    return yhat_linear


def evaluate_accuracy_scratch(data_iterator, net, model_params, model_ctx):
    numerator = 0.
    denominator = 0.
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        output = net(data, model_params)
        predictions = nd.argmax(output, axis=1)
        numerator += nd.sum(predictions == label)
        denominator += data.shape[0]
    return (numerator / denominator).asscalar()
