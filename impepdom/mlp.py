import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MultilayerPerceptron(nn.Module):
    def __init__(self, input_size=294, num_hidden_layers=2, hidden_layer_size=20):
        '''
        Initialize an `num_hidden_layers + 2` neural network.

        Parameters
        ----------
        input_size: int
            Length of input vector
        
        num_hidden_layers: int
            Number of equivalent hidden layers

        hidden_layer_size: int
            Number of neurons in each hidden layer
        '''

        super(MultilayerPerceptron, self).__init__()     
        self.inp_sz = input_size
        self.num_hid = num_hidden_layers
        self.hid_sz = hidden_layer_size
        self.hidden = nn.ModuleList()  # initialize list of layers
        self.dropout = nn.ModuleList()  # initialize list of dropout-able layers

        self.dropout.append(nn.Dropout(p=0.85))  # dropout for input layer
        self.hidden.append(nn.Linear(input_size, hidden_layer_size))  # first hidden layer
        for _ in range(1, num_hidden_layers):
            self.dropout.append(nn.Dropout(p=0.65))  # dropout for hidden layers
            self.hidden.append(nn.Linear(hidden_layer_size, hidden_layer_size))  # fully-connected hidden layers
        self.hidden.append(nn.Linear(hidden_layer_size, 1))  # output layer
        
    def forward(self, x):
        '''
        Feed-forward for the network. 

        Parameters
        ----------
        x: ndarray
            Input vector of size `self.input_size`
        '''

        for i in range(self.num_hid):
            x = F.relu(self.dropout[i](self.hidden[i](x)))

        x = torch.sigmoid(self.hidden[-1](x))  # classification output
        
        return x

    def get_my_name(self):
        return "mlp"