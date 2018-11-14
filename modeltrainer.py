'''
modeltrainer.py

This file parses through the enron_cleaned.csv and inputs a training set of each subject and body (as tensors) into the LSTM network for email subject generation. It then runs the test set and compares the results (a bag of highly probable words) with the actual subject for manual evaluation.
'''

import networkmaker
import pandas as pd
import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import sys

import random

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# helper method to find the index of a letter in the alphabet
def letter_to_index(letter):
    if letter not in alphabet:
        print(letter)
    return alphabet.find(letter)

# returns a list of 1 x 27 tensors for each input word
def word_to_tensor_list(word):
    tensor = torch.zeros(len(word), 1, len(alphabet))
    for li, letter in enumerate(word):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor

# old stopwords list
# stopwords = ['a', 'an', 'the', 'and', 'or', 'of', 'for', 'to', 'in', 'from', 'not', 'but', 'up']

## used to filter out stopwords while training models so their losses will not be included
stopwords = ['a', 'an', 'the', 'and', 'or', 'of', 'for', 'to', 'in', 'from', 'not', 'but', 'up', 'a','b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u','v', 'w', 'x', 'w', 'z', 'y', 'wa', 'was', 'i', 'you', 'they', 'on', 'among', 'as', 'they', 'theirs', 'their', 'them', 'we', 'us', 'ours', 'our', 'she', 'he', 'her', 'him', 'me', 'my', 'mine', 'been', 'be', 'wont', 'would', 'wouldnt', 'must', 'have', 'has', 'had', 'will', 'about', 'out', 'which', 'what', 'why', 'when', 'where', 'how', 'are', 'is', 'am', 'were', 'may', 'out', 'enron', 'dont', 'do', 'did', 'didnt', 'if', 'need', 'this', 'that', 'these', 'those', 'it', 'about', 'with', 'so', 'at', 'by', 'bb', 're', 'forward', 'fyi', 'im', 'ho', 'thats', 'ok', 'subject', 'you', 'your', 'yours']

alphabet = 'abcdefghijklmnopqrstuvwxyz '

# stopword tensor to input into networkmaker.py
stopwords_tensor = [word_to_tensor_list(word + " ") for word in stopwords]
generator_fw = networkmaker.SubjectGenerator(stopwords_tensor) # forward-input network
generator_bw = networkmaker.SubjectGenerator(stopwords_tensor) # backward-input network

# arbitrarily decided testing set
test = [2, 23, 3, 56, 999, 444, 333, 787, 387, 2223, 566, 974, 2100, 2345, 1786, 1973, 389, 26, 5, 667]
'''
Remove the stopwords:
important parameter to control for during email subject generation
'''
def remove_stopwords(words):
    result = []
    for word in words:
        if word not in stopwords:
            result.append(word)

    return result

# plot the loss functions
def plot_loss(fw_loss_train, bw_loss_train, nepochs):
    flr = plt.plot(range(nepochs), fw_loss_train, color = 'blue')
    #wlr = plt.plot(range(nepochs), fw_loss_test, color = 'orange')
    flt = plt.plot(range(nepochs), bw_loss_train, color = 'blue', linestyle = '--')
    #wlt = plt.plot(range(nepochs), bw_loss_test, color = 'orange', linestyle = '--')
    '''
    legend1 = plt.legend(handles = [mlines.Line2D([], [], color = 'blue', label = 'train'), mlines.Line2D([], [], color = 'orange', label = 'test')], bbox_to_anchor=(1, 1))
    '''
    legend2 = plt.legend(handles = [mlines.Line2D([], [], label = 'forward', color = 'blue'), mlines.Line2D([], [], linestyle = '--', label = 'backward', color = 'blue')])
    x = plt.xlabel("Epoch")
    y = plt.ylabel("Loss")
    title = plt.title("Loss in each epoch by forward and backward models and by test and training sets")
    plt.savefig("loss_graph.png")

def makeSubjectTensor(word_list, subject_list):
    word_tensor_list = []
    for word in word_list:
        if word in subject_list:
            word_tensor_list.append(torch.FloatTensor([[1]]))
        else:
            word_tensor_list.append(torch.FloatTensor([[0]]))
    return word_tensor_list

'''
Read in the data and create test and traing sets; train the SubjectGenerator Model
using the training set and predict the test set email bodies using the model
'''
def main():
    df = pd.read_csv("enron_cleaned.csv", index_col = 0)


    nepochs = 20
    loss_fw_train_list = []
    loss_bw_train_list = []
    loss_fw_test_list = []
    loss_bw_test_list = []
    
    # train data
    # loop through each subject and body in training set and create a list of tensors
    for n in range(nepochs):
        
        loss_fw_train = 0
        loss_bw_train = 0
        loss_fw_test = 0
        loss_bw_test = 0
        
        for index, row in df.iterrows():
            
            # only process training set
            if index in test:
                continue
            
            subject = row['Subject']
            body = row['Body']

            # divide into lists of words
            body_list = body.split()
            subject_list = subject.split()

            # remove stopwords (approach 1)
            #body_list = remove_stopwords(body_list)
            #subject_list = remove_stopwords(subject_list)
            b_tensor_list = []
            s_tensor_list = []

            if len(body_list) == 0 or len(subject_list) == 0:
                continue

            # for every word, add empty space and create a list of tensors for it
            for w_b in body_list:
                b_tensor_list.append(word_to_tensor_list(w_b + " "))

            s_tensor_list = makeSubjectTensor(body_list, subject_list)


            ## pass into the network
            # forward pass
            loss_fw_train += generator_fw.train_pattern(b_tensor_list, s_tensor_list)

            # backward pass
            loss_bw_train += generator_bw.train_pattern(b_tensor_list[::1], s_tensor_list[::1])
    
                
        loss_fw_train_list.append(loss_fw_train)
        loss_bw_train_list.append(loss_bw_train)
        
        # must create saved dir in the current directory
        generator_fw.save_model("./saved/fw" + str(n))
        generator_bw.save_model("./saved/bw" + str(n))
        print("Epoch", n, "is finished!")
    
    print("loss list for training test for forward model is", ','.join(map(str, loss_fw_train_list)))
    print("loss list for training test for backward model is", ','.join(map(str, loss_bw_train_list)))
    
    plot_loss(loss_fw_train_list, loss_bw_train_list, nepochs)
    
    # save everything to test.txt in the directory
    train_existing_model(df, generator_fw, generator_bw, "./test.txt")


def train_existing_model(df, fw, bw, file):

    f = open(file, "w")
    # run and examine the results of the test set
    for t in test:
        subject_t = df.iloc[t]['Subject']
        body_t = df.iloc[t]['Body']
        
        body_list_t = body_t.split()
        subject_list_t = subject_t.split()
        
        # stopwords removal
        body_list_t = remove_stopwords(body_list_t)
        subject_list_t = remove_stopwords(subject_list_t)
        
        test_tensor_list = []
        
        for w_b_t in body_list_t:
            test_tensor_list.append(word_to_tensor_list(w_b_t + " "))
    
        # get the output tensor for the particular subject after feeding it into the network
        # forward
        output_subject_fw = fw.eval_pattern(test_tensor_list) # returns a scalar of probabilities
        
        # backward
        output_subject_bw = bw.eval_pattern(test_tensor_list) # returns a scalar of probabilities
        
        # results of test set can be used as results of the network
        body_t = ' '.join(body_list_t)
        subject_t = ' '.join(subject_list_t)
        
        print("The body of the email is: \n")
        print(body_t)
        print()
        
        f.write("The body of the email is: \n")
        f.write(body_t)
        f.write("\n")
        
        print("The actual subject of the email is: \n")
        print(subject_t)
        print()
        
        f.write("The actual subject of the email is: \n")
        f.write(subject_t)
        f.write("\n")
        
        str_out_fw = []
        str_out_bw = []
        # convert output tensor probabilities to Python list format
        for i in range(len(output_subject_fw)):
            str_out_fw.append(output_subject_fw[i].item())
        
        for i in range(len(output_subject_bw)):
            str_out_bw.append(output_subject_bw[i].item())
        
        # update probability list by taking the maximum of all occurences of every unique word
        unique_body_list = []
        max_out_fw = []
        max_out_bw = []
        max_out = []
        visited = []

        for word in body_list_t:
            # skip the word if it is not unique and has been visited
            if word in visited:
                continue
            indices = [i for i, x in enumerate(body_list_t) if x == word]
            unique_body_list.append(word)
            max_fw = max([str_out_fw[i] for i in range(len(body_list_t)) if i in indices])
            max_bw = max([str_out_bw[i] for i in range(len(body_list_t)) if i in indices])
            max_out_fw.append(max_fw)
            max_out_bw.append(max_bw)
            max_out.append(max(max_bw, max_fw))
            visited.append(word)

        # return list of strings for each model sorted by the corresponding probabilities
        sorted_str_fw = [x for _,x in sorted(zip(max_out_fw, unique_body_list))]
        sorted_str_bw = [x for _,x in sorted(zip(max_out_bw, unique_body_list))]
        sorted_str_max = [x for _,x in sorted(zip(max_out, unique_body_list))]
        
        
        print("All words in body sorted by activations for forward model: \n")
        print(','.join(sorted_str_fw))
        print()
        
        f.write("All words in body sorted by activations for forward model: \n")
        f.write(','.join(sorted_str_fw))
        f.write("\n")
        
        print("All words in body sorted by activations for backward model: \n")
        print(','.join(sorted_str_bw))
        print()
        
        f.write("All words in body sorted by activations for backward model: \n")
        f.write(','.join(sorted_str_bw))
        f.write("\n")

        print("All words in body sorted by activations for maximum model: \n")
        print(','.join(sorted_str_max))
        print()
        
        f.write("All words in body sorted by activations for maximum model: \n")
        f.write(','.join(sorted_str_max))
        f.write("\n")
        
        print("Activations for all words in forward model: \n")
        print(','.join(map(str, max_out_fw)))
        print()
        
        f.write("Activations for all words in forward model: \n")
        f.write(','.join(map(str, max_out_fw)))
        f.write("\n")
        
        print("Activations for all words in backward model: \n")
        print(','.join(map(str, max_out_bw)))
        print()
        
        f.write("Activations for all words in backward model: \n")
        f.write(','.join(map(str, max_out_bw)))
        f.write("\n")
            
        print("Activations for all words in maximum model: \n")
        print(','.join(map(str, max_out)))
        print()
    
        f.write("Activations for all words in maximum model: \n")
        f.write(','.join(map(str, max_out)))
        f.write("\n")

    f.close()

# load existing model and test with testing set 
# otherwise, run main() and train and test in the same process
args = sys.argv
if "--load" in args:
    index = args.index("--load")
    d = args[index+1]
    epochs = int(args[index+2])
    df = pd.read_csv("enron_cleaned.csv", index_col = 0)
    file = args[index + 3]
    fw = networkmaker.load_model(d + "/fw" + str(epochs-1))
    bw = networkmaker.load_model(d + "/bw" + str(epochs-1))
    train_existing_model(df, fw, bw, file)
else:
    main()
