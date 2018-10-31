import networkmaker
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

## implementing stopwords cleaning
 # stoplist = ['a', 'an', 'the', 'and', 'or', 'of', 'for', 'to', 'in', 'from', 'not', 'but', 'up']
 # maybe also add 'http' and '.com'?

def remove_stopwords(words):
    return words

def list_to_tensor_list(words):
    tensor = torch.zeros(len(words), 1, len(alphabet))
    for li, letter in enumerate(words):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor

def letter_to_index(letter):
    if letter not in alphabet:
        print(letter)
    return alphabet.find(letter)

alphabet = 'abcdefghijklmnopqrstuvwxyz '

generator = networkmaker.SubjectGenerator()
df = pd.read_csv("enron_cleaned.csv", index_col = 0)
print(df.head(5))


# loop through each subject and body and feed into the network
for index, row in df.iterrows():
    subject = row['Subject']
    body = row['Body']

    # get the tensor representation of the list
    s_tensor_list = list_to_tensor_list(subject)
    b_tensor_list = list_to_tensor_list(body)
    
    # pass into the network
    
