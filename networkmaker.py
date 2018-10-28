import torch
import torch.nn as nn

class SubjectGenerator(nn.Module):
    """a class for our model"""

    def __init__(self):
        """creates a new model with our architecture etc."""
        super(SubjectGenerator, self).__init__()
        inputSize = 27 # all the letters and space
        hidden1size = 100 # LSTM node count
        hidden2size = 50 # fully connected node count
        outputSize = 1 # output

        self.LSTMLayer = nn.LSTMCell(inputSize, hidden1size)
        self.relu1 = nn.ReLU()
        self.fullyConnectedLayer = nn.Linear(hidden1size, hidden2size)
        self.relu2 = nn.ReLU()
        self.outputLayer = nn.Linear(hidden2size, outputSize)
        self.outputActivation = nn.Sigmoid()

    def forward(self, character, hidden, cell):
        """a forward pass through the network, for predicting and training"""
        newhid, newcell = self.LSTMLayer(character, (hidden,cell))
        newhid = self.relu1(newhid)
        newhid = self.fullyConnectedLayer(newhid)
        newhid = self.relu2(newhid)
        newhid = self.outputLayer(newhid)
        newhid = self.outputActivation(newhid)
        return newhid, newcell

    def blank_cell_and_hidden(self):
        """returns empty cells and hidden for the first pass"""
        return torch.zeroes(1,self.hidden1size), torch.zeroes(1,self.hidden1size)

    def save_model(self,destination):
        """saves the model so we don't have to retrain"""
        torch.save(self.state_dict(), destination)
