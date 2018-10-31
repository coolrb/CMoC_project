import torch
import torch.nn as nn

class SubjectGenerator(nn.Module):
    """a class for our model"""

    def __init__(self):
        """creates a new model with our architecture etc."""
        super(SubjectGenerator, self).__init__()

        self.learning_rate = .001

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

        self.blank_cell_and_hidden()

        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate) # stochastic gradient descent for optimization
        self.criterion = nn.NLLLoss() # log-likelihood loss function


    def forward_one_letter(self, character):
        """a forward pass through the network that uses just one letter
        We want the network to process data on a per-letter basis but update
        its loss on a per-word basis so we're doing it this way"""
        newhid, newcell = self.LSTMLayer(character, (self.hidden_carry,self.cell_carry))
        newhid = self.relu1(newhid)
        newhid = self.fullyConnectedLayer(newhid)
        newhid = self.relu2(newhid)
        newhid = self.outputLayer(newhid)
        newhid = self.outputActivation(newhid)
        self.hidden_carry, self.cell_carry = newhid, newcell

    def forward(self, word):
        """a forward pass through the network, for predicting and training"""
        for character in word:
            forward_one_letter(self, character)

    def blank_cell_and_hidden(self):
        """returns empty cells and hidden for the first pass"""
        self.hidden_carry = torch.zeroes(1,self.hidden1size), torch.zeroes(1,self.hidden1size)
        self.cell_carry = torch.zeroes(1,self.hidden1size), torch.zeroes(1,self.hidden1size)

    def save_model(self,destination):
        """saves the model so we don't have to retrain"""
        torch.save(self.state_dict(), destination)

    def train_pattern(self, body, subject_key):
        '''
        trains the network on a single message based on the subject_key
        subject line should be a list of 0's and 1's depending on if the corresponding word is in the subject line of the message
        note: MESSAGES are encoded the as a list of WORDS
        WORDS are encoded as a list of CHARACTERS
        CHARACTERS are encoded as 27-dimensional 1-hot-vectors
        '''
        self.train()
        self.zero_grad()
        loss = 0
        self.blank_cell_and_hidden()
        for i in range(len(body)):
            self(body[i])
            loss += self.criterion(self.hidden_carry(), subject_key[i])
        loss.backward()
        self.optimizer.step()
        return loss.data.numpy() / len(subject_key)

    def eval_pattern(self, body):
        """evaluate novel body text"""
        self.eval()
        self.blank_cell_and_hidden()
        for i in range(len(body)):
            self(body[i])
        return self.hidden_carry

def load_model(source):
    """loads model from state dict so we don't have to retrain"""
    checkpoint = torch.load('models/convnet_mnist_pretrained_3layer_arch.pt')
    model = Net()
    model.load_state_dict(checkpoint)
    return model
