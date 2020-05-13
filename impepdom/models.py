import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MultilayerPerceptron(nn.Module):
    def __init__(self, input_size=308, num_hidden_layers=2, hidden_layer_size=100, dropout_input=0.85, dropout_hidden=0.65, convolutional=False, num_conv_hidden_layers=1, filter_size=5, stride=2):
        '''
        Initialize an `num_hidden_layers + 2` neural network.

        Parameters
        ----------
        input_size: int
            Length of input vector, should be NUM_AA * max_aa_len
        
        num_hidden_layers: int
            Number of equivalent hidden layers

        hidden_layer_size: int
            Number of neurons in each hidden layer

        dropout_input: float

        dropout_hidden: float

        convolutional: True/False

        num_conv
        '''

        super(MultilayerPerceptron, self).__init__()     
        self.inp_sz = input_size
        self.num_hid = num_hidden_layers
        self.hid_sz = hidden_layer_size
        self.hidden = nn.ModuleList()  # initialize list of layers
        self.dropout = nn.ModuleList()  # initialize list of dropout-able layers

        self.dropout[0] = nn.Dropout(p=dropout_input)  # dropout for input layer
        self.hidden.append(nn.Linear(input_size, hidden_layer_size))  # first hidden layer
        for i in range(1, num_hidden_layers):
            self.dropout[i] = nn.Dropout(p=dropout_hidden)  # dropout for hidden layers
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
        name = "mlp_{0}x{1}".format(self.num_hid, self.hid_sz) 
        return name


    
