# coding=utf-8
# coding=utf-8
from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd, gluon

from DL_gluon.common_utils import transform


ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
data_ctx = ctx
model_ctx = ctx


class MLP(gluon.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        with self.name_scope():
            self.dense0 = gluon.nn.Dense(64)
            self.dense1 = gluon.nn.Dense(64)
            self.dense2 = gluon.nn.Dense(10)

    def forward(self, x):
        x = nd.relu(self.dense0(x))
        print("Hidden Representation 1: %s" % x)
        x = nd.relu(self.dense1(x))
        print("Hidden Representation 1: %s" % x)
        x = self.dense2(x)
        print("Network output: %s" % x)
        return x


def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]


if __name__ == "__main__":
    num_inputs = 784
    num_outputs = 10
    num_example = 60000
    batch_size = 64

    train_data = mx.gluon.data.DataLoader(gluon.data.vision.MNIST(train=True, transform=transform),
                                          batch_size, shuffle=True)
    test_data = mx.gluon.data.DataLoader(gluon.data.vision.MNIST(train=False, transform=transform),
                                         batch_size, shuffle=False)

    ## first method
    # net = MLP()
    # net.collect_params().initialize(mx.init.Normal(sigma=0.01), ctx=model_ctx)
    #
    # data = nd.ones((1, 784))
    # net(data.as_in_context(model_ctx))

    ## second method
    number_hidden = 64
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(number_hidden, activation='relu'))
        net.add(gluon.nn.Dense(number_hidden, activation='relu'))
        net.add(gluon.nn.Dense(num_outputs))

    net.collect_params().initialize(mx.init.Normal(sigma=0.1), ctx=model_ctx)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})

    epochs = 10
    smoothing_constant = 0.01
    for e in range(epochs):
        cumulative_loss = 0
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(model_ctx).reshape((-1, 784))
            label = label.as_in_context(model_ctx)
            with autograd.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(data.shape[0])
            cumulative_loss += nd.sum(loss).asscalar()

        test_accuracy = evaluate_accuracy(test_data, net)
        train_accuracy = evaluate_accuracy(train_data, net)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
              (e, cumulative_loss / num_example, train_accuracy, test_accuracy))
