# coding=utf-8
from __future__ import print_function

import matplotlib.pyplot as plt

import mxnet as mx
from mxnet import nd, autograd, gluon
mx.random.seed(1)

data_ctx = mx.cpu()
model_ctx = mx.cpu()


def real_fn(X):
    return 2 * X[:, 0] - 3.4 * X[:, 1] + 4.2


def synthetic_data(number_inputs, number_outputs, number_samples):
    X = nd.random_normal(shape=(num_samples, number_inputs), ctx=data_ctx)
    noise = .1 * nd.random_normal(shape=(number_samples, ), ctx=data_ctx)
    y = real_fn(X) + noise
    return X, y


def plot_raw_data(X, y):
    plt.scatter(X[:, 1].asnumpy(), y.asnumpy())
    plt.show()


def net(input_data, weight, b):
    # neural networks
    return mx.nd.dot(input_data, weight) + b


def square_loss(y_est, y):
    return nd.mean((y_est - y) ** 2)


def SGD(model_params, learning_rate):
    for param in model_params:
        param[:] = param - learning_rate * param.grad
    return model_params


def plot_loss(losses, X, weight, b, sample_size=100):
    xs = list(range(len(losses)))
    f, (fg1, fg2) = plt.subplots(1, 2)
    fg1.set_title("Loss during training")
    fg1.plot(xs, losses, '-r')
    fg2.set_title("Estimated vs real function")
    fg2.plot(X[:sample_size, 1].asnumpy(), net(X[:sample_size, :], weight, b).asnumpy(), 'or', label='Estimated')
    fg2.plot(X[:sample_size, 1].asnumpy(), real_fn(X[:sample_size, :]).asnumpy(), '*g', label="Real")
    fg2.legend()
    plt.show()


if __name__ == "__main__":
    num_inputs = 2
    num_outputs = 1
    num_samples = 10000

    X, y = synthetic_data(num_inputs, num_outputs, num_samples)
    print([X[0], y[0]])

    # data iterators
    batch_size = 4
    train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X, y), batch_size=batch_size, shuffle=True)

    for i, (data, label) in enumerate(train_data):
        print(data, label)
        break

    # allocate memories for model parameters
    w = nd.random_normal(shape=(num_inputs, num_outputs), ctx=model_ctx)
    b = nd.random_normal(shape=num_outputs, ctx=model_ctx)
    params = [w, b]

    # allocate memories for each gradient
    for param in params:
        param.attach_grad()

    # batch training
    epochs = 10
    learning_rate = 0.0001
    num_batches = num_samples / batch_size
    losses = []
    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(model_ctx)
            label = label.as_in_context(model_ctx).reshape((-1, 1))
            with autograd.record():
                output = net(data, w, b)
                loss = square_loss(output, label)
            loss.backward()
            SGD(params, learning_rate)
            cumulative_loss += loss.asscalar()
        print(cumulative_loss / num_batches)
        losses.append(cumulative_loss / num_batches)

    plot_loss(losses, X, w, b)