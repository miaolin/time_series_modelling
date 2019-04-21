# coding=utf-8
from __future__ import print_function
import mxnet as mx
import numpy as np
from mxnet import nd, autograd
from DL_gluon.chapter_4.common_utils import transform
from DL_gluon.common_utils import relu, softmax, softmax_cross_entropy, SGD
from DL_gluon.chapter_3.common_utils import evaluate_accuracy_scratch
mx.random.seed(1)
ctx = mx.cpu()


def pure_batch_norm(X, gamma, beta, eps=1e-5):
    if len(X.shape) not in (2, 4):
        raise ValueError("only supports dense or 2dconv")

    if len(X.shape) == 2:
        mean = nd.mean(X, axis=0)
        variance = nd.mean((X - mean) ** 2, axis=0)
        X_hat = (X - mean) * 1.0 / nd.sqrt(variance + eps)
        out = gamma * X_hat + beta

    elif len(X.shape) == 4:
        N, C, H, W = X.shape
        mean = nd.mean(X, axis=(0, 2, 3))
        variance = nd.mean((X - mean.reshape((1, C, 1, 1))) ** 2, axis=(0, 2, 3))
        X_hat = (X - mean.reshape((1, C, 1, 1))) * 1.0 / nd.sqrt(variance.reshape(1, C, 1, 1) + eps)
        out = gamma.reshape((1, C, 1, 1)) * X_hat
    return out


def batch_norm(X, gamma, beta, momentum=0.9, eps=1e-5, scope_name="", is_training=True, debug=False):

    global _BN_MOVING_MEANS, _BN_MOVING_VARS

    # usual batch normmalization
    if len(X.shape) not in (2, 4):
        raise ValueError("only supports dense or 2dconv")

    if len(X.shape) == 2:
        mean = nd.mean(X, axis=0)
        variance = nd.mean((X - mean) ** 2, axis=0)
        if is_training:
            X_hat = (X - mean) * 1.0 / nd.sqrt(variance + eps)
        else:
            X_hat = (X - _BN_MOVING_MEANS[scope_name]) * 1.0 / nd.sqrt(_BN_MOVING_VARS[scope_name] + eps)
        out = gamma * X_hat + beta

    elif len(X.shape) == 4:
        N, C, H, W = X.shape
        mean = nd.mean(X, axis=(0, 2, 3))
        variance = nd.mean((X - mean.reshape((1, C, 1, 1))) ** 2, axis=(0, 2, 3))
        if is_training:
            X_hat = (X - mean.reshape((1, C, 1, 1))) * 1.0 / nd.sqrt(variance.reshape(1, C, 1, 1) + eps)
        else:
            X_hat = (X - _BN_MOVING_MEANS[scope_name].reshape((1, C, 1, 1))) * 1.0 / \
                    nd.sqrt(_BN_MOVING_VARS[scope_name].reshape((1, C, 1, 1)) + eps)
        out = gamma.reshape((1, C, 1, 1)) * X_hat + beta.reshape((1, C, 1, 1))

    # keep the moving statistics
    try:
        _BN_MOVING_MEANS, _BN_MOVING_VARS
    except:
        _BN_MOVING_MEANS, _BN_MOVING_VARS = {}, {}

    if scope_name not in _BN_MOVING_MEANS:
        _BN_MOVING_MEANS[scope_name] = mean
    else:
        _BN_MOVING_MEANS[scope_name] = _BN_MOVING_MEANS[scope_name] * momentum + mean * (1.0 - momentum)

    if scope_name not in _BN_MOVING_VARS:
        _BN_MOVING_VARS[scope_name] = variance
    else:
        _BN_MOVING_VARS[scope_name] = _BN_MOVING_VARS[scope_name] * momentum + variance * (1.0 - momentum)

    if debug:
        print('== info start ==')
        print('scope_name = {}'.format(scope_name))
        print('mean = {}'.format(mean))
        print('var = {}'.format(variance))
        print('_BN_MOVING_MEANS = {}'.format(_BN_MOVING_MEANS[scope_name]))
        print('_BN_MOVING_VARS = {}'.format(_BN_MOVING_VARS[scope_name]))
        print('output = {}'.format(out))
        print('== info end ==')
    return out


def net(X, params, is_training=True, debug=False):

    W1, b1, gamma1, beta1, W2, b2, gamma2, beta2, W3, b3, gamma3, beta3, W4, b4 = params

    h1_conv = nd.Convolution(data=X, weight=W1, bias=b1, kernel=(3, 3), num_filter=20)
    h1_normed = batch_norm(h1_conv, gamma1, beta1, scope_name='bn1', is_training=is_training)
    h1_activation = relu(h1_normed)
    h1 = nd.Pooling(data=h1_activation, pool_type="avg", kernel=(2, 2), stride=(2, 2))
    if debug:
        print("h1 shape: %s" % (np.array(h1.shape)))

    h2_conv = nd.Convolution(data=h1, weight=W2, bias=b2, kernel=(5, 5), num_filter=50)
    h2_normed = batch_norm(h2_conv, gamma2, beta2, scope_name='bn2', is_training=is_training)
    h2_activation = relu(h2_normed)
    h2 = nd.Pooling(data=h2_activation, pool_type="avg", kernel=(2, 2), stride=(2, 2))
    if debug:
        print("h2 shape: %s" % (np.array(h2.shape)))

    h2 = nd.flatten(h2)
    if debug:
        print("Flat h2 shape: %s" % (np.array(h2.shape)))

    h3_linear = nd.dot(h2, W3) + b3
    h3_normed = batch_norm(h3_linear, gamma3, beta3, scope_name="bn3", is_training=is_training)
    h3 = relu(h3_normed)

    yhat_linear = nd.dot(h3, W4) + b4
    if debug:
        print("yhat_linear shape: %s" % (np.array(yhat_linear.shape)))
    return yhat_linear


def test_norm():
    ga = nd.array([1, 1], ctx=ctx)
    be = nd.array([0, 0], ctx=ctx)

    A = nd.array([1, 7, 5, 4, 6, 10], ctx=ctx).reshape((3, 2))
    print(A)

    norm_A = pure_batch_norm(A, gamma=ga, beta=be)
    print(norm_A)

    B = nd.array([1, 6, 5, 7, 4, 3, 2, 5, 6, 3, 2, 4, 5, 3, 2, 5, 6], ctx=ctx).reshape((2, 2, 2, 2))
    print(B)
    norm_B = pure_batch_norm(B, ga, be)
    print(norm_B)


if __name__ == "__main__":
    batch_size = 64
    num_inputs = 784
    num_outputs = 10

    train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform),
                                          batch_size, shuffle=True)
    test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                                         batch_size, shuffle=False)

    test_norm()

    weight_scale = 0.01
    num_fc = 128

    W1 = nd.random_normal(shape=(20, 1, 3, 3), scale=weight_scale, ctx=ctx)
    b1 = nd.random_normal(shape=20, scale=weight_scale, ctx=ctx)

    gamma1 = nd.random_normal(shape=20, loc=1, scale=weight_scale, ctx=ctx)
    beta1 = nd.random_normal(shape=20, scale=weight_scale, ctx=ctx)

    W2 = nd.random_normal(shape=(50, 20, 5, 5), scale=weight_scale, ctx=ctx)
    b2 = nd.random_normal(shape=50, scale=weight_scale, ctx=ctx)

    gamma2 = nd.random_normal(shape=50, loc=1, scale=weight_scale, ctx=ctx)
    beta2 = nd.random_normal(shape=50, scale=weight_scale, ctx=ctx)

    W3 = nd.random_normal(shape=(800, num_fc), scale=weight_scale, ctx=ctx)
    b3 = nd.random_normal(shape=num_fc, scale=weight_scale, ctx=ctx)

    gamma3 = nd.random_normal(shape=num_fc, loc=1, scale=weight_scale, ctx=ctx)
    beta3 = nd.random_normal(shape=num_fc, scale=weight_scale, ctx=ctx)

    W4 = nd.random_normal(shape=(num_fc, num_outputs), scale=weight_scale, ctx=ctx)
    b4 = nd.random_normal(shape=10, scale=weight_scale, ctx=ctx)
    params = [W1, b1, gamma1, beta1, W2, b2, gamma2, beta2, W3, b3, gamma3, beta3, W4, b4]

    for param in params:
        param.attach_grad()

    # test run
    for data, _ in train_data:
        data = data.as_in_context(ctx)
        break
    output = net(data, params, is_training=True, debug=True)

    # training
    # this training needs gpu, using cpu would be very slow
    epochs = 1
    moving_loss = 0.0
    learning_rate = 0.01
    for e in range(epochs):
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            label_one_hot = nd.one_hot(label, num_outputs)
            with autograd.record():
                output = net(data, params, is_training=True)
                loss = softmax_cross_entropy(output, label_one_hot)
            loss.backward()
            SGD(params, learning_rate)

            if i == 0:
                moving_loss = nd.mean(loss).asscalar()
            else:
                moving_loss = 0.99 * moving_loss + 0.01 * nd.mean(loss).asscalar()

        test_accuracy = evaluate_accuracy_scratch(test_data, net, model_params=params, model_ctx=ctx)
        train_accuracy = evaluate_accuracy_scratch(train_data, net, model_params=params, model_ctx=ctx)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))
