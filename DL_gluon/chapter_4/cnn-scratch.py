# coding=utf-8
from __future__ import print_function
import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon
from DL_gluon.common_utils import relu, softmax, softmax_cross_entropy, SGD


ctx = mx.cpu()
mx.random.seed(1)


def transform(data, label):
    return nd.transpose(data.astype(np.float32), (2, 0, 1))/255, label.astype(np.float32)


def evaluate_accuracy(data_iterator, net, params, num_filter_layer1, num_filter_layer2):
    numerator = 0.
    denominator = 0.
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data, params, num_filter_layer1, num_filter_layer2)
        predictions = nd.argmax(output, axis=1)
        numerator += nd.sum(predictions == label)
        denominator += data.shape[0]
    return (numerator / denominator).asscalar()


def net(X, params, num_filter_layer1, num_filter_layer2, debug=False):
    # params = [W1, b1, W2, b2, W3, b3, W4, b4]
    # computation of the first convolutional layer
    h1_conv = nd.Convolution(data=X, weight=params[0], bias=params[1], kernel=(3, 3),
                             num_filter=num_filter_layer1)
    h1_activation = relu(h1_conv)
    h1 = nd.Pooling(data=h1_activation, pool_type="avg", kernel=(2, 2), stride=(2, 2))
    if debug:
        print("h1 shape: %s" % (np.array(h1.shape)))

    # define the computation of the second convolutional layer
    h2_conv = nd.Convolution(data=h1, weight=params[2], bias=params[3], kernel=(5, 5),
                             num_filter=num_filter_layer2)
    h2_activation = relu(h2_conv)
    h2 = nd.Pooling(data=h2_activation, pool_type="avg", kernel=(2, 2), stride=(2, 2))
    if debug:
        print("h2 shape: %s" % (np.array(h2.shape)))

    # flattening h2 so that we can feed it into a fully-connected layer
    h2 = nd.flatten(h2)
    if debug:
        print("flat h2 shape: %s" % (np.array(h2.shape)))

    # define the computation of the third fully-connected layer
    h3_linear = nd.dot(h2, params[4]) + params[5]
    h3 = relu(h3_linear)
    if debug:
        print("flat h3 shape: %s" % (np.array(h3.shape)))

    yhat_linear = nd.dot(h3, params[6]) + params[7]
    if debug:
        print("yhat_linear shape: %s" % (np.array(yhat_linear.shape)))

    return yhat_linear


if __name__ == "__main__":
    batch_size = 64
    num_inputs = 784
    num_outputs = 10

    train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform),
                                          batch_size, shuffle=True)
    test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                                         batch_size, shuffle=False)

    weight_scale = 0.01
    num_fc = 128
    num_filter_conv_layer1 = 20
    num_filter_conv_layer2 = 50

    W1 = nd.random_normal(shape=(num_filter_conv_layer1, 1, 3, 3), scale=weight_scale, ctx=ctx)
    b1 = nd.random_normal(shape=num_filter_conv_layer1, scale=weight_scale, ctx=ctx)

    W2 = nd.random_normal(shape=(num_filter_conv_layer2, num_filter_conv_layer1, 5, 5),
                          scale=weight_scale, ctx=ctx)
    b2 = nd.random_normal(shape=num_filter_conv_layer2, scale=weight_scale, ctx=ctx)

    W3 = nd.random_normal(shape=(800, num_fc), scale=weight_scale, ctx=ctx)
    b3 = nd.random_normal(shape=num_fc, scale=weight_scale, ctx=ctx)

    W4 = nd.random_normal(shape=(num_fc, num_outputs), scale=weight_scale, ctx=ctx)
    b4 = nd.random_normal(shape=num_outputs, scale=weight_scale, ctx=ctx)

    params = [W1, b1, W2, b2, W3, b3, W4, b4]
    for param in params:
        param.attach_grad()

    for data, _ in train_data:
        data = data.as_in_context(ctx)
        break
    print(data.shape)
    print(W1.shape)
    conv = nd.Convolution(data=data, weight=W1, bias=b1, kernel=(3, 3), num_filter=num_filter_conv_layer1)
    print(conv.shape)

    # average pooling
    pool = nd.Pooling(data=conv, pool_type="max", kernel=(2, 2), stride=(2, 2))
    print(pool.shape)

    # test run
    output = net(data, params, num_filter_layer1=num_filter_conv_layer1,
                 num_filter_layer2=num_filter_conv_layer2, debug=True)

    # training loop
    epochs = 5
    learning_rate = 0.01
    smoothing_constant = 0.01
    moving_loss = 0.0
    for e in range(epochs):
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            label_one_hot = nd.one_hot(label, num_outputs)
            with autograd.record():
                output = net(data, params, num_filter_layer1=num_filter_conv_layer1,
                             num_filter_layer2=num_filter_conv_layer2)
                loss = softmax_cross_entropy(output, label_one_hot)
            loss.backward()
            SGD(params, learning_rate)

            # keep a moving average loss
            curr_loss = nd.mean(loss).asscalar()
            if i == 0 and e == 0:
                moving_loss = curr_loss
            else:
                moving_loss = (1 - smoothing_constant) * moving_loss + smoothing_constant *curr_loss

        test_accuracy = evaluate_accuracy(test_data, net, params, num_filter_conv_layer1, num_filter_conv_layer2)
        train_accuracy = evaluate_accuracy(train_data, net, params, num_filter_conv_layer1, num_filter_conv_layer2)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))