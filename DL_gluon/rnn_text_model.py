# coding=utf-8
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn, rnn


class RNNModel(gluon.Block):
    """A model with an encode, recurrent layer, and a decoder."""

    def __int__(self, mode, vocab_size, num_embed, num_hidden, num_layers,
                dropout=0.5, tie_weights=False, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        with self.name_scope():
            self.drop = nn.Dropout(dropout)
            self.encoder = nn.Embedding(vocab_size, num_embed,
                                        weight_initializer=mx.init.Uniform(0, 1))

            if mode == 'rnn_relu':
                self.rnn = rnn.RNN(num_hidden, num_layers, activation='relu', dropout=dropout, input_size=num_embed)
            elif mode == 'rnn_tanh':
                self.rnn = rnn.RNN(num_hidden, num_layers, dropout=dropout, input_size=num_embed)
            elif mode == 'lstm':
                self.rnn = rnn.LSTM(num_hidden, num_layers, dropout=dropout, input_size=num_embed)
            elif mode == 'gru':
                self.rnn = rnn.GRU(num_hidden, num_layers, dropout=dropout, input_size=num_embed)
            else:
                raise ValueError("Invalid mode {}. Options are rnn_relu, rnn_tanh, lstm, and gru".format(mode))

            if tie_weights:
                self.decoder = nn.Dense(vocab_size, in_units=num_hidden, params=self.encoder.params)
            else:
                self.decoder = nn.Dense(vocab_size, in_units=num_hidden)
            self.num_hidden = num_hidden

    def forward(self, inputs, hidden):
        emb = self.drop(self.encoder(inputs))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.reshape(-1, self.num_hidden))
        return decoded, hidden

    def begin_state(self, *args, **kwards):
        return self.rnn.begin_state(*args, **kwards)