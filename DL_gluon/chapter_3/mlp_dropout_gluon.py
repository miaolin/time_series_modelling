# coding=utf-8
from __future__ import print_function
import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon
from DL_gluon.common_utils import transform
from DL_gluon.chapter_3.common_utils import evaluate_accuracy

ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()


if __name__ == "__main__":
    batch_size = 64
    num_inputs = 784
    num_outputs = 10

    train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform),
                                          batch_size, shuffle=True)
    test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                                         batch_size, shuffle=False)

    # define the model
    num_hidden = 256
    net = gluon.nn.Sequential()
    with net.name_scope():
        # add first hidden layer
        net.add(gluon.nn.Dense(num_hidden, activation="relu"))

        # add a dropout
        net.add(gluon.nn.Dropout(.5))

        # add second hidden layer
        net.add(gluon.nn.Dense(num_hidden, activation="relu"))

        # add a dropout
        net.add(gluon.nn.Dropout(.5))

        net.add(gluon.nn.Dense(num_outputs))

    net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)

    for x, _ in train_data:
        x = x.as_in_context(ctx)
        break
    print(net(x[0:1]))
    print(net(x[0:1]))

    # result should be the same
    with autograd.predict_mode():
        print(net(x[0:1]))
        print(net(x[0:1]))

    # result should be different
    with autograd.train_mode():
        print(net(x[0:1]))
        print(net(x[0:1]))

    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {"learning_rate": 0.1})

    # training loop
    epochs = 10
    smoothing_constant = 0.01
    moving_loss = 0.0
    for e in range(epochs):
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(ctx).reshape((-1, 784))
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)
                loss.backward()
            trainer.step(data.shape[0])

            curr_loss = nd.mean(loss).asscalar()
            if (i == 0) and (e == 0):
                moving_loss = curr_loss
            else:
                moving_loss = (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss

        test_accuracy = evaluate_accuracy(test_data, net, ctx)
        train_accuracy = evaluate_accuracy(train_data, net, ctx)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))
