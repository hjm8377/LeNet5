import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        
        self.C1 = nn.Conv2d(1, 6, 5)
        self.tanh1 = nn.Tanh()
        self.S2 = nn.AvgPool2d(2, 2)
        self.C3 = nn.Conv2d(6, 16, 5)
        self.tanh2 = nn.Tanh()
        self.S4 = nn.AvgPool2d(2, 2)
        self.C5 = nn.Conv2d(16, 120, 5)
        self.tanh3 = nn.Tanh()
        self.flatten = nn.Flatten()
        self.F6 = nn.Linear(120, 84)
        self.tanh4 = nn.Tanh()
        self.OutputLayer = nn.Linear(84,10)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.C1(x)
        x = self.tanh1(x)
        x = self.S2(x)
        x = self.C3(x)
        x = self.tanh2(x)
        x = self.S4(x)
        x = self.C5(x)
        x = self.tanh3(x)
        x = self.flatten(x)
        x = self.F6(x)
        x = self.tanh4(x)
        x = self.OutputLayer(x)
        x = self.softmax(x)
        
        return x
