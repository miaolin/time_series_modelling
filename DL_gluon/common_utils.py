# coding=utf-8


def SGD(model_params, learning_rate):
    for param in model_params:
        param[:] = param - learning_rate * param.grad
    return model_params