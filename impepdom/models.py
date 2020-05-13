import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ALL_AA = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'U', 'X']
NUM_AA = len(ALL_AA)  # number of amino acids (21 + 1 unknown)

class MultilayerPerceptron(nn.Module):
    def __init__(self, input_size=308, num_hidden_layers=2, hidden_layer_size=100, dropout_input=0.85, dropout_hidden=0.65, conv=False, num_conv_layers=1, conv_filt_sz=5, conv_stride=2):
        '''
        Initialize an `num_hidden_layers + 2` neural network.

        Parameters
        ----------
        input_size: int
            Length of input vector, should be NUM_AA * max_pep_len
        
        num_hidden_layers: int
            Number of equivalent hidden layers

        hidden_layer_size: int
            Number of neurons in each hidden layer

        dropout_input: float
            Dropout rate of the input layer

        dropout_hidden: float
            Dropout rate of the non-convolutional hidden layers

        conv: True/False
            Whether convolutional layers are added

        num_conv_layers: int
            Number of convolutional layers

        conv_filt_sz: int
            Size of the filter (the number of amino acids)

        conv_stride: int
            Gap between filters
        '''

        super(MultilayerPerceptron, self).__init__()     
        self.inp_sz = input_size
        self.num_hid = num_hidden_layers
        self.hid_sz = hidden_layer_size
        self.num_conv_layers = num_conv_layers
        self.conv_filt_sz = conv_filt_sz
        self.conv_stride = conv_stride
        self.hidden = nn.ModuleList()  # initialize list of layers
        self.conv = nn.ModuleList()
        self.dropout = nn.ModuleList()  # initialize list of dropout-able layers

        # With convolution
        if conv:
            self.conv.append(nn.Conv1d(in_channels=1, out_channels=24, kernel_size=conv_filt_sz*NUM_AA, stride=conv_stride*NUM_AA, padding=(conv_filt_sz-1)*NUM_AA))
            self.conv.append(nn.MaxPool1d(kernel_size=2))
            for i in range(1, num_conv_layers):
                self.conv.append(nn.Conv1d(in_channels=24, out_channels=24, kernel_size=conv_filt_sz*NUM_AA, stride=conv_stride*NUM_AA, padding=(conv_filt_sz-1)*NUM_AA))
                self.conv.append(nn.MaxPool1d(kernel_size=2))

        self.dropout.append(nn.Dropout(p=dropout_input))  # dropout for input layer
        self.hidden.append(nn.Linear(48, hidden_layer_size))  # first hidden layer !!! FIX THIS LATER (number of input neurons) !!!
        for _ in range(1, num_hidden_layers):
            self.dropout.append(nn.Dropout(p=dropout_hidden))  # dropout for hidden layers
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
        
        for j in range(self.num_conv_layers*2)[::2]:
            print(j)
            x = F.relu(self.conv[j](x))
            x = self.conv[j+1](x)
        
        x = x.flatten()
        print(x.size())
            
        for i in range(self.num_hid):
            x = F.relu(self.dropout[i](self.hidden[i](x)))

        x = torch.sigmoid(self.hidden[-1](x))  # classification output
        
        return x

    def get_my_name(self):
        name = "mlp_{0}x{1}".format(self.num_hid, self.hid_sz) 
        return name
