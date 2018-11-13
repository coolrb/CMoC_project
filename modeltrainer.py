'''
modeltrainer.py

This file parses through the enron_cleaned.csv and inputs a training set of each subject and body (as tensors)
into the LSTM network for email subject generation. It then runs the test set and compares the results 
(a bag of highly probable words) with the actual subject for manual evaluation.
'''

import networkmaker
import pandas as pd
import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import random

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def letter_to_index(letter):
    if letter not in alphabet:
        print(letter)
    return alphabet.find(letter)

def word_to_tensor_list(word):
    tensor = torch.zeros(len(word), 1, len(alphabet))
    for li, letter in enumerate(word):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor

## implementing stopwords cleaning
stopwords = ['a', 'an', 'the', 'and', 'or', 'of', 'for', 'to', 'in', 'from', 'not', 'but', 'up', 'a','b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u','v', 'w', 'x', 'w', 'z', 'y', 'wa', 'was', 'i', 'you', 'they', 'on', 'among', 'as', 'they', 'theirs', 'their', 'them', 'we', 'us', 'ours', 'our', 'she', 'he', 'her', 'him', 'me', 'my', 'mine', 'been', 'be', 'wont', 'would', 'wouldnt', 'must', 'have', 'has', 'had', 'will', 'about', 'out', 'which', 'what', 'why', 'when', 'where', 'how', 'are', 'is', 'am', 'were', 'may', 'out', 'enron', 'dont', 'do', 'did', 'didnt', 'if', 'need', 'this', 'that', 'these', 'those', 'it', 'about', 'with', 'so', 'at', 'by', 'bb', 're', 'forward', 'fyi', 'im', 'ho', 'thats', 'ok', 'subject', 'you', 'your', 'yours']
 # maybe also add 'http' and '.com'?
alphabet = 'abcdefghijklmnopqrstuvwxyz '
threshold = .5 # to determine if words should be included in the subject or not

stopwords_tensor = [word_to_tensor_list(word + " ") for word in stopwords]
generator_fw = networkmaker.SubjectGenerator(stopwords_tensor) # forward-input network
generator_bw = networkmaker.SubjectGenerator(stopwords_tensor) # backward-input network

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


def scalar_to_word(scalar, subject_list):
    result_list = []
    for i in range(len(scalar)): # assuming scalar is a list
        prob = scalar[i]
        if prob > threshold:
            result_list.append(subject_list[i])

    return result_list

# might not be needed
def tensor_list_to_word(tensor_list):
    pass

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

    num_rows = len(df.index)

    # specify the number of training and testing sets
    num_test = 20
    num_train = num_rows - num_test

    test = random.sample(range(num_rows), num_test)
    nepochs = 10
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

            subject = row['Subject']
            body = row['Body']

            # divide into lists of words
            body_list = body.split()
            subject_list = subject.split()

            # remove stopwords
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

            #for w_s in subject_list:
                #s_tensor_list.append(word_to_tensor_list(w_s + " "))

            ## pass into the network
            if index not in test:
                # forward pass
                loss_fw_train += generator_fw.train_pattern(b_tensor_list, s_tensor_list)

                # backward pass
                loss_bw_train += generator_bw.train_pattern(b_tensor_list[::1], s_tensor_list[::1])
            '''
            else:
                # forward pass
                loss_fw_test += generator_fw.train_pattern(b_tensor_list, s_tensor_list)

                # backward pass
                loss_bw_test += generator_bw.train_pattern(b_tensor_list[::1], s_tensor_list[::1])
            '''
                
        loss_fw_train_list.append(loss_fw_train)
        loss_bw_train_list.append(loss_bw_train)

        generator_fw.save_model("./saved/fw" + str(n))
        generator_bw.save_model("./saved/bw" + str(n))
        print("Epoch", n, "is finished!")
    
    print("loss list for training test for forward model is", ','.join(map(str, loss_fw_train_list)))
    print("loss list for training test for backward model is", ','.join(map(str, loss_bw_train_list)))
    
    plot_loss(loss_fw_train_list, loss_bw_train_list, nepochs)

    f = open("./test.txt", "w")
    # run and examine the results of the test set
    for t in test:
        subject_t = df.iloc[t]['Subject']
        body_t = df.iloc[t]['Body']
        
        body_list_t = body_t.split()
        subject_list_t = subject_t.split()
        
        # stopwords removal
        #body_list_t = remove_stopwords(body_list_t)
        #subject_list_t = remove_stopwords(subject_list_t)
        
        test_tensor_list = []
        
        for w_b_t in body_list_t:
            test_tensor_list.append(word_to_tensor_list(w_b_t + " "))
    
        # get the output tensor for the particular subject after feeding it into the network
        # forward
        output_subject_fw = generator_fw.eval_pattern(test_tensor_list) # returns a scalar of probabilities
        #predicted_subject_fw = scalar_to_word(output_subject_fw, body_list_t)
        
        # backward
        output_subject_bw = generator_bw.eval_pattern(test_tensor_list) # returns a scalar of probabilities
        #predicted_subject_bw = scalar_to_word(output_subject_bw, body_list_t)
        
        #TODO: write all of these to a text file
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
        for i in range(len(output_subject_fw)):
            str_out_fw.append(output_subject_fw[i].item())
        
        for i in range(len(output_subject_bw)):
            str_out_bw.append(output_subject_bw[i].item())
        
        sorted_str_fw = [x for _,x in sorted(zip(str_out_fw, body_list_t))]
        sorted_str_bw = [x for _,x in sorted(zip(str_out_bw, body_list_t))]
        
        
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
        
        
        
        
        print("Activations for all words in forward model: \n")
        print(','.join(map(str, str_out_fw)))
        print()
        
        f.write("Activations for all words in forward model: \n")
        f.write(','.join(map(str, str_out_fw)))
        f.write("\n")
        
        print("Activations for all words in backward model: \n")
        print(','.join(map(str, str_out_bw)))
        print()
        
        f.write("Activations for all words in backward model: \n")
        f.write(','.join(map(str, str_out_bw)))
        f.write("\n")

    f.close()

main()
