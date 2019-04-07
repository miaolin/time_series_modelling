# coding=utf-8
from __future__ import print_function
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
import matplotlib.pyplot as plt
mx.random.seed(1)

data_ctx = mx.cpu()
model_ctx = mx.cpu()

def transform(data, label):
    return data.astype(np.float32) / 255, label.astype(np.float32)


def show_image(image):
    plt.imshow(image.asnumpy())
    plt.show()


def softmax(y_linear):
    exp = nd.exp(y_linear - nd.max(y_linear, axis=1).reshape((-1, 1)))
    norms = nd.sum(exp, axis=1).reshape((-1, 1))
    return exp / norms


def net(X, Weight, bias):
    y_linear = nd.dot(X, Weight) + bias
    yhat = softmax(y_linear)
    return yhat


def cross_entropy(yhat, y):
    return - nd.sum(y * nd.log(yhat + 1e-6))


def SGD(model_params, lr):
    for param in model_params:
        param[:] = param - lr * param.grad
    return params


def evaluate_accuracy(data_iterator, net, Weight, bias):
    numerator = 0.
    denominator = 0.
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        #label_one_hot = nd.one_hot(label, 10)
        output = net(data, Weight, bias)
        predictions = nd.argmax(output, axis=1)
        numerator += nd.sum(predictions == label)
        denominator += data.shape[0]
    return (numerator / denominator).asscalar()


def model_prediction(net, data, weight, bias):
    output = net(data, weight, bias)
    return nd.argmax(output, axis=1)


if __name__ == "__main__":

    mnist_train = gluon.data.vision.MNIST(train=True, transform=transform)
    mnist_test = gluon.data.vision.MNIST(train=False, transform=transform)
    image, label = mnist_train[0]
    print(image.shape, label)

    num_inputs = 784
    num_outputs = 10
    num_example = 60000

    im = mx.nd.tile(image, (1, 1, 3))
    print(im.shape)

    #show_image(im)

    batch_size = 64
    train_data = mx.gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
    test_data = mx.gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)

    W = nd.random_normal(shape=(num_inputs, num_outputs), ctx=model_ctx)
    b = nd.random_normal(shape=num_outputs, ctx=model_ctx)
    params = [W, b]

    for param in params:
        param.attach_grad()

    # check softmax function
    # sample_y_linear = nd.random_normal(shape=(2, 10))
    # sample_yhat = softmax(sample_y_linear)
    # print(sample_yhat)

    print(evaluate_accuracy(test_data, net, W, b))

    # training
    epochs = 10
    learning_rate = 0.005
    for e in range(epochs):
        cumulative_loss = 0.0
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(model_ctx).reshape((-1, 784))
            label = label.as_in_context(model_ctx)
            label_one_hot = nd.one_hot(label, 10)
            with autograd.record():
                output = net(data, W, b)
                loss = cross_entropy(output, label_one_hot)
            loss.backward()
            SGD(params, learning_rate)
            cumulative_loss += nd.sum(loss).asscalar()

        test_accuracy = evaluate_accuracy(test_data, net, W, b)
        train_arracy = evaluate_accuracy(train_data, net, W, b)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, cumulative_loss/num_example, train_arracy,
                                                                 test_accuracy))


    # sample 10 random data for testing
    sample_data = mx.gluon.data.DataLoader(mnist_test, 10, shuffle=True)
    for i, (data, label) in enumerate(sample_data):
        data = data.as_in_context(model_ctx)
        print(data.shape)
        im = nd.transpose(data, (1, 0, 2, 3))
        im = nd.reshape(im, (28, 10*28, 1))
        imtitles = nd.tile(im, (1, 1, 3))

        plt.imshow(imtitles.asnumpy())
        plt.show()
        pred = model_prediction(net, data.reshape((-1, 784)), W, b)
        print('model predictions are: ', pred)
        break
