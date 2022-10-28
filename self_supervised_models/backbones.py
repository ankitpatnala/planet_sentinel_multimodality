import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self,
            input_dim,
            num_layers,
            hidden_dim,
            dropout=0):
        super(MLP,self).__init__()

        self.layers = []
        for i in range(num_layers-1):
            self.layers.append(nn.Sequential(nn.Linear(input_dim if i==0 else hidden_dim,hidden_dim),
                            nn.ReLU(),
                            nn.Dropout(dropout)))
        self.layers.append(nn.Linear(hidden_dim,hidden_dim))

        self.mlp = nn.Sequential(*self.layers)
    
    def forward(self,x):
        x = self.mlp(x)
        return x


class ResMLP(nn.Module):
    def __init__(self,
            input_dim,
            num_layers,
            hidden_dim,
            dropout=0.0):
        super(ResMLP,self).__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.initial_layer =  nn.Linear(input_dim,hidden_dim)
        self.resnet_blocks = []
        for i in range(self.num_layers):
            self.resnet_blocks.append(Resnet_block(self.hidden_dim,self.dropout))
        self.resnet_module = nn.Sequential(*self.resnet_blocks)
        self.prediction_layer = nn.Sequential(
                                    nn.BatchNorm1d(self.hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_dim,self.hidden_dim))

    def forward(self,x):
        x = self.initial_layer(x)
        x = self.resnet_module(x)
        x = self.prediction_layer(x)
        return x

class Resnet_block(nn.Module):
    def __init__(self,hidden_dim,dropout):
        super(Resnet_block,self).__init__()
        self.model = nn.Sequential(
                nn.BatchNorm1d(hidden_dim),
                nn.Linear(hidden_dim,hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim,hidden_dim),
                nn.Dropout(dropout))
    
    def forward(self,x):
        return x + self.model(x)

        



