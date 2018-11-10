import torch
import torch.nn as nn

class SubjectGenerator(nn.Module):
    """a class for our model"""

    def __init__(self, stopwords):
        """creates a new model with our architecture etc."""
        super(SubjectGenerator, self).__init__()


        self.stopwords = stopwords
        self.learning_rate = .001

        self.inputSize = 27 # all the letters and space
        self.hidden1size = 10 # LSTM node count
        self.hidden2size = 5 # fully connected node count
        self.outputSize = 1 # output

        self.LSTMLayer = nn.LSTMCell(self.inputSize, self.hidden1size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout()
        self.fullyConnectedLayer = nn.Linear(self.hidden1size, self.hidden2size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout()
        self.outputLayer = nn.Linear(self.hidden2size, self.outputSize)
        self.outputActivation = nn.Sigmoid()

        self.blank_cell_and_hidden()

        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate) # stochastic gradient descent for optimization
        self.criterion = nn.BCELoss() # log-likelihood loss function


    def forward_one_letter(self, character, is_final=False):
        """a forward pass through the network that uses just one letter
        We want the network to process data on a per-letter basis but update
        its loss on a per-word basis so we're doing it this way"""
        newhid, newcell = self.LSTMLayer(character, (self.hidden_carry,self.cell_carry))
        newhid = self.relu1(newhid)
        newcell = self.relu1(newcell)
        newhid = self.dropout1(newhid)
        newcell = self.dropout1(newcell)

        self.hidden_carry, self.cell_carry = newhid, newcell
        if is_final:
            newhid = self.fullyConnectedLayer(newhid)
            newhid = self.relu2(newhid)
            newhid = self.dropout2(newhid)
            newhid = self.outputLayer(newhid)
            newhid = self.outputActivation(newhid)
            return newhid

    def forward(self, word):
        """a forward pass through the network, for predicting and training"""
        for character in range(len(word)):
            if character == len(word)-1:
                return self.forward_one_letter(word[character], is_final=True)
            self.forward_one_letter(word[character])

    def blank_cell_and_hidden(self):
        """resets empty cells and hidden for the first pass"""
        self.hidden_carry = torch.zeros(1,self.hidden1size)
        self.cell_carry = torch.zeros(1,self.hidden1size)

    def save_model(self,destination):
        """saves the model so we don't have to retrain"""
        torch.save(self.state_dict(), destination)

    def train_pattern(self, body, subject_key):
        '''
        trains the network on a single message based on the subject_key
        subject_key should be a list of 0's and 1's depending on if the corresponding word is in the subject line of the message
        note: MESSAGES are encoded the as a list of WORDS
        WORDS are encoded as a list of CHARACTERS (ending with space)
        CHARACTERS are encoded as 27-dimensional 1-hot-vectors
        '''
        self.train()
        self.zero_grad()
        loss = 0
        self.blank_cell_and_hidden()
        for i in range(len(body)):
            newpred = self(body[i])
            in_stopwords = False
            for word_to_check in self.stopwords:
                is_this_word = len(body[i]) == len(word_to_check)
                if is_this_word:
                    for j,letter_to_check in enumerate(word_to_check):
                        if not (letter_to_check.equal(body[i][j])):
                            is_this_word = False
                            break
                if is_this_word:
                    in_stopwords = True
                    break
            if not in_stopwords:
                loss += self.criterion(newpred, subject_key[i])
        print("Loss:",loss)
        print("Words:",i)
        print("Loss per word:",loss/i)
        loss.backward()
        self.optimizer.step()
        return loss.data.numpy() / len(subject_key)

    def eval_pattern(self, body):
        """evaluate novel body text"""
        self.eval()
        self.blank_cell_and_hidden()
        newpreds = []
        for i in range(len(body)):
            newpreds.append(self(body[i]))
        return newpreds

def load_model(source):
    """loads model from state dict so we don't have to retrain"""
    checkpoint = torch.load('models/convnet_mnist_pretrained_3layer_arch.pt')
    model = Net()
    model.load_state_dict(checkpoint)
    return model
