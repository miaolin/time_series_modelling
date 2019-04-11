# coding=utf-8
from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd, gluon

from DL_gluon.chapter_3.common_utils import net, evaluate_accuracy_scratch
from DL_gluon.common_utils import transform, softmax_cross_entropy, SGD


ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
data_ctx = ctx
model_ctx = ctx

if __name__ == "__main__":
    num_inputs = 784
    num_outputs = 10
    num_example = 60000
    batch_size = 64

    train_data = mx.gluon.data.DataLoader(gluon.data.vision.MNIST(train=True, transform=transform),
                                          batch_size, shuffle=True)
    test_data = mx.gluon.data.DataLoader(gluon.data.vision.MNIST(train=False, transform=transform),
                                         batch_size, shuffle=False)

    # allocation parameters
    num_hidden = 256
    weight_scale = 0.01

    # first layer
    W1 = nd.random_normal(shape=(num_inputs, num_hidden), scale=weight_scale, ctx=model_ctx)
    b1 = nd.random_normal(shape=num_hidden, scale=weight_scale, ctx=model_ctx)

    W2 = nd.random_normal(shape=(num_hidden, num_hidden), scale=weight_scale, ctx=model_ctx)
    b2 = nd.random_normal(shape=num_hidden, scale=weight_scale, ctx=model_ctx)

    W3 = nd.random_normal(shape=(num_hidden, num_outputs), scale=weight_scale, ctx=model_ctx)
    b3 = nd.random_normal(shape=num_outputs, scale=weight_scale, ctx=model_ctx)

    params = [W1, b1, W2, b2, W3, b3]

    for param in params:
        param.attach_grad()

    epochs = 10
    learning_rate = 0.001
    smoothing_constant = 0.01
    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(model_ctx).reshape((-1, 784))
            label = label.as_in_context(model_ctx)
            label_one_hot = nd.one_hot(label, 10)
            with autograd.record():
                output = net(data, params)
                loss = softmax_cross_entropy(output, label_one_hot)
            loss.backward()
            SGD(params, learning_rate)
            cumulative_loss += nd.sum(loss).asscalar()

        test_accuracy = evaluate_accuracy_scratch(test_data, net, params, model_ctx)
        train_accuracy = evaluate_accuracy_scratch(train_data, net, params, model_ctx)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
              (e, cumulative_loss / num_example, train_accuracy, test_accuracy))
