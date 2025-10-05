import torch
import torch.nn as nn

class SingleHiddenLayerNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim,
                 hidden_activation='tanh', output_activation='linear'):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hidden_activation_name = hidden_activation
        self.output_activation_name = output_activation

        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

        if hidden_activation == 'tanh':
            self.hidden_act = nn.Tanh()
        elif hidden_activation == 'relu':
            self.hidden_act = nn.ReLU()
        elif hidden_activation == 'sigmoid':
            self.hidden_act = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown hidden activation: {hidden_activation}")

        if output_activation == 'linear':
            self.output_act = nn.Identity()
        elif output_activation == 'sigmoid':
            self.output_act = nn.Sigmoid()
        elif output_activation == 'softmax':
            self.output_act = nn.Softmax(dim=1)
        else:
            raise ValueError(f"Unknown output activation: {output_activation}")

    def forward(self, x):
        h = self.hidden_act(self.hidden(x))
        out = self.output_act(self.output(h))
        return out