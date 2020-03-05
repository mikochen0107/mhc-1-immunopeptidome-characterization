import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MultilayerPerceptron(nn.Module):
    def __init__(self):
        super(MultilayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(1,2)
        self.fc2 = nn.Linear(2,1)
        
    def forward(self, x):
        x = self.fc2(F.relu(self.fc1(x)))
        return x