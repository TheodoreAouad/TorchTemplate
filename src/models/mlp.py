import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, layers, *args, **kwargs):
        super().__init__()

        self.layers = layers
        self.mlp = None

        self._builder(*args, **kwargs)

    def _builder(self, *args, **kwargs):
        dense_layers = []
        for idx, neurons in enumerate(self.layers[:-1], 1):
            dense_layers.append(nn.Linear(neurons, self.layers[idx], *args, **kwargs))
        self.mlp = nn.Sequential(*dense_layers)

    def forward(self, x, activation=lambda x: x, *args, **kwargs):
        output = x
        for layer in self.mlp:
            output = activation(layer(output))
        return output
