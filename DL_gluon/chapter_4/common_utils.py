# coding=utf-8
import numpy as np
from mxnet import nd


def transform(data, label):
    return nd.transpose(data.astype(np.float32), (2, 0, 1))/255, label.astype(np.float32)