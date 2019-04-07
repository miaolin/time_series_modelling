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
    X = nd.random_normal(shape=(number_samples, number_inputs), ctx=data_ctx)
    noise = .1 * nd.random_normal(shape=(number_samples, ), ctx=data_ctx)
    y = real_fn(X) + noise
    return X, y


if __name__ == "__main__":
    num_inputs = 2
    num_outputs = 1
    num_samples = 10000

    X, y = synthetic_data(num_inputs, num_outputs, num_samples)

    # data iterators
    batch_size = 4
    train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X, y), batch_size=batch_size, shuffle=True)

    # define the network
    net = gluon.nn.Dense(1, in_units=2)
    print(net.weight)
    print(net.bias)

    net.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=model_ctx)
    square_loss = gluon.loss.L2Loss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.0001})

    # training loop
    epochs = 10
    loss_sequence = []
    num_batches = num_samples / batch_size
    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(model_ctx)
            label = label.as_in_context(model_ctx)
            with autograd.record():
                output = net(data)
                loss = square_loss(output, label)
            loss.backward()
            trainer.step(batch_size)
            cumulative_loss += nd.mean(loss).asscalar()
        print("Epoch %s, loss: %s" % (e, cumulative_loss / num_samples))
        loss_sequence.append(cumulative_loss)

    # parameters
    params = net.collect_params()
    for param in params.values():
        print(param.name, param.data())