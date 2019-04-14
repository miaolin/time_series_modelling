# coding=utf-8
from __future__ import print_function
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon

from DL_gluon.chapter_4.common_utils import transform

mx.random.seed(1)
ctx = mx.cpu()


def evaluate_accuracy(data_iterator, net, ctx):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]


if __name__ == "__main__":
    batch_size = 64
    num_inputs = 784
    num_outputs = 10

    train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform),
                                          batch_size, shuffle=True)
    test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                                         batch_size, shuffle=False)

    #num_fc = 512
    num_fc = 128
    net = gluon.nn.Sequential()
    with net.name_scope():
        #net.add(gluon.nn.Conv2D(channels=20, kernel_size=3, activation="relu"))
        net.add(gluon.nn.Conv2D(channels=20, kernel_size=5, activation="relu"))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        net.add(gluon.nn.Conv2D(channels=50, kernel_size=5, activation="relu"))
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))

        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.Dense(num_fc, activation="relu"))
        net.add(gluon.nn.Dense(num_outputs))

    net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .1})

    epochs = 5
    smoothing_constant = 0.01
    moving_loss = 0.0
    for e in range(epochs):
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(data.shape[0])

            curr_loss = nd.mean(loss).asscalar()
            if i == 0 and e == 0:
                moving_loss = curr_loss
            else:
                moving_loss = (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss

        test_accuracy = evaluate_accuracy(test_data, net, ctx)
        train_accuracy = evaluate_accuracy(train_data, net, ctx)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))
