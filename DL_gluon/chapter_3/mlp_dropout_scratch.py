# coding=utf-8
from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd, gluon

from DL_gluon.common_utils import transform, relu, softmax_cross_entropy, SGD, dropout
from DL_gluon.chapter_3.common_utils import net_dropout, evaluate_accuracy_scratch


ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
data_ctx = ctx
model_ctx = ctx


if __name__ == "__main__":

    # test dropout
    A = nd.arange(20).reshape((5, 4))
    print(dropout(A, 0.0))

    print(dropout(A, 0.5))

    print(dropout(A, 1.0))

    num_inputs = 784
    num_outputs = 10
    num_example = 60000
    batch_size = 64

    mnist = mx.test_utils.get_mnist()
    train_data = mx.gluon.data.DataLoader(gluon.data.vision.MNIST(train=True, transform=transform),
                                          batch_size, shuffle=True)
    test_data = mx.gluon.data.DataLoader(gluon.data.vision.MNIST(train=False, transform=transform),
                                         batch_size, shuffle=False)

    W1 = nd.random_normal(shape=(784, 256), ctx=ctx) * 0.01
    b1 = nd.random_normal(shape=256, ctx=ctx) * 0.01
    W2 = nd.random_normal(shape=(256, 128), ctx=ctx) * 0.01
    b2 = nd.random_normal(shape=128, ctx=ctx) * 0.01
    W3 = nd.random_normal(shape=(128, 10), ctx=ctx) * 0.01
    b3 = nd.random_normal(shape=10, ctx=ctx) * 0.01

    params = [W1, b1, W2, b2, W3, b3]
    for param in params:
        param.attach_grad()

    epochs = 10
    moving_loss = 0.0
    learning_rate = 0.001
    for e in range(epochs):
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(ctx).reshape((-1, 784))
            label = label.as_in_context(ctx)
            label_one_hot = nd.one_hot(label, 10)
            with autograd.record():
                output = net_dropout(data, params, drop_prob=0.5)
                loss = softmax_cross_entropy(output, label_one_hot)
            loss.backward()
            SGD(params, learning_rate)

            if i == 0:
                moving_loss = nd.mean(loss).asscalar()
            else:
                moving_loss = 0.99 * moving_loss + 0.01 * nd.mean(loss).asscalar()

        test_accuracy = evaluate_accuracy_scratch(test_data, net_dropout, params, model_ctx)
        train_accuracy = evaluate_accuracy_scratch(train_data, net_dropout, params, model_ctx)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy,
                                                                 test_accuracy))