
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        # 80, 180, 256, 348
        self.fc1 = nn.Linear(816, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 2)
        #  self.fc4 = nn.Linear(180, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, bsz):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_size, num_classes)
        self.hidden_size = hidden_size
        self.bsz = bsz
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (Variable(torch.zeros(self.num_layers, 1, self.hidden_size)),
                Variable(torch.zeros(self.num_layers, 1, self.hidden_size)))

    def forward(self, x):
        # print type(x), x
        print self.hidden
        out, self.hidden = self.lstm(x)
        print "hidden: ", self.hidden
        print "out: ", out
        tag_space = self.hidden2tag(out)
        return F.log_softmax(tag_space)
