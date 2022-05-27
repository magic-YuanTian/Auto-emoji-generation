from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import torch.utils.data as Data
import math
import copy


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x_test, y_test in loader:
            x_test = x_test.to(device=device)
            y_test = y_test.to(device=device)

            scores = model(x_test)
            _, predictions = scores.max(1) # get the first value at dim=1 (batc_size x class_num)
            num_correct += (predictions == y_test).sum()
            num_samples += predictions.size(0)

        acc = float(num_correct)/float(num_samples)

    model.train()
    return acc


# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print("device in MyModels.py: ", device)

# NN
class myNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(myNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        return out


# RNN
class myRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(myRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # N x time_seq x features
        self.fc = nn.Linear(hidden_size, num_classes)  # capture the last hidden state

    # N x time_seq x features
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward Prop
        out, _ = self.rnn(x, h0)  # 1. output state  2. hidden state
        out = out[:, -1, :]  # only take the last hidden state
        out = self.fc(out)
        return out


# GRU
class myGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(myGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # N x time_seq x features
        self.fc = nn.Linear(hidden_size, num_classes)  # capture the last hidden state

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward Prop
        out, _ = self.gru(x, h0)  # 1. output state  2. hidden state
        out = out[:, -1, :]  # only take the last hidden state
        out = self.fc(out)
        return out


# LSTM
class myLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(myLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # N x time_seq x features
        self.fc = nn.Linear(hidden_size, num_classes)  # capture the last hidden state

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward Prop
        out, _ = self.lstm(x, (h0, c0))  # 1. output state  2. hidden state
        out = out[:, -1, :]  # only take the last hidden state
        out = self.fc(out)
        # softmax_layer = nn.Softmax(dim=1)
        # out = softmax_layer(out)
        return out


# LSTM captured all hidden state
class myLSTMall(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, sequence_length):
        super(myLSTMall, self).__init__()
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # N x time_seq x features
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)  # capture the last hidden state

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward Prop
        out, _ = self.lstm(x, (h0, c0))  # 1. output state  2. hidden state
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out

# Bi-LSTM
class myBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(myBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # N x time_seq x features
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # capture the last hidden state

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        # Forward Prop
        out, _ = self.lstm(x, (h0, c0))  # 1. output state  2. hidden state
        out = out[:, -1, :]  # only take the last hidden state
        out = self.fc(out)
        return out

