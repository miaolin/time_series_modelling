# coding=utf-8
import mxnet as mx
from mxnet import nd, autograd, gluon
import matplotlib.pyplot as plt


data_ctx = mx.cpu()
model_ctx = mx.cpu()


def logistic(z):
    return 1.0 / (1.0 + nd.exp(-z))


def process_data(raw_data):
    train_lines = raw_data.splitlines()
    num_examples = len(train_lines)
    num_features = 123
    X = nd.zeros((num_examples, num_features), ctx=data_ctx)
    Y = nd.zeros((num_examples, 1), ctx=data_ctx)
    for i, line in enumerate(train_lines):
        tokens = line.split()
        label = (int(tokens[0]) + 1) / 2  # Change label from {-1,1} to {0,1}
        Y[i] = label
        for token in tokens[1:]:
            index = int(token[:-2]) - 1
            X[i, index] = 1
    return X, Y


def log_loss(output, y):
    yhat = logistic(output)
    return - nd.nansum(y * nd.log(yhat) + (1 - y) * nd.log(1 - yhat))


if __name__ == "__main__":
    with open("../../data/adult/a1a.train") as f:
        train_raw = f.read()

    with open("../../data/adult/a1a.test") as f:
        test_raw = f.read()

    Xtrain, Ytrain = process_data(train_raw)
    Xtest, Ytest = process_data(test_raw)

    print(Xtrain.shape)
    print(Ytrain.shape)
    print(Xtest.shape)
    print(Ytest.shape)

    print(nd.sum(Ytrain) / len(Ytrain))
    print(nd.sum(Ytest) / len(Ytest))

    batch_size = 64
    train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(Xtrain, Ytrain),
                                       batch_size=batch_size, shuffle=True)
    test_data = gluon.data.DataLoader(gluon.data.ArrayDataset(Xtest, Ytest),
                                      batch_size=batch_size, shuffle=True)

    # define the model
    net = gluon.nn.Dense(1)
    net.collect_params().initialize(mx.init.Normal(sigma=.1), ctx=model_ctx)

    # instantiated an optimizer
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})

    epoches = 30
    loss_sequence = []
    num_examples = len(Xtrain)

    for e in range(epoches):
        cumulative_loss = 0.0
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(model_ctx)
            label = label.as_in_context(model_ctx)
            with autograd.record():
                output = net(data)
                loss = log_loss(output, label)
            loss.backward()
            trainer.step(batch_size)
            cumulative_loss += nd.sum(loss).asscalar()
        print("Epoch %s, loss: %s" % (e, cumulative_loss))
        loss_sequence.append(cumulative_loss)
