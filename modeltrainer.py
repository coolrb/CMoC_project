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

## implementing stopwords cleaning
stopwords = ['a', 'an', 'the', 'and', 'or', 'of', 'for', 'to', 'in', 'from', 'not', 'but', 'up']
 # maybe also add 'http' and '.com'?
alphabet = 'abcdefghijklmnopqrstuvwxyz '
threshold = .5 # to determine if words should be included in the subject or not
generator = networkmaker.SubjectGenerator()

'''
Remove the stopwords:
important parameter to control for during email subject generation
'''
def remove_stopwords(words):
    '''
    result = []
    for word in words:
        if word not in stopwords:
            result.append(word)

    return result
    '''
    return words

def word_to_tensor_list(word):
    tensor = torch.zeros(len(word), 1, len(alphabet))
    for li, letter in enumerate(word):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor

def letter_to_index(letter):
    if letter not in alphabet:
        print(letter)
    return alphabet.find(letter)

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

    # train data
    # loop through each subject and body in training set and create a list of tensors
    for index, row in df.iterrows():
        if index in test:
            continue

        subject = row['Subject']
        body = row['Body']

        # divide into lists of words
        body_list = body.split()
        subject_list = subject.split()

        # remove stopwords
        body_list = remove_stopwords(body_list)
        subject_list = remove_stopwords(subject_list)

        b_tensor_list = []
        s_tensor_list = []

        # for every word, add empty space and create a list of tensors for it
        for w_b in body_list:
            b_tensor_list.append(word_to_tensor_list(w_b + " "))

        for w_s in subject_list:
            s_tensor_list.append(word_to_tensor_list(w_s + " "))
        
        ## pass into the network

        # forward pass
        #generator.train_pattern(body = b_tensor_list, subject_line = s_tensor_list)

        # backward pass
        #generator.train_pattern(body = b_tensor_list.reverse(), subject_line = s_tensor_list.reverse())

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
        output_subject = generator.eval_pattern(test_tensor_list) # returns a scalar of probabilities
        predicted_subject = scalar_to_word(output_subject, subject_i.split())
        
        # results of test set can be used as results of the network
        print("The body of the email is: \n")
        print(body_i)
        print()
        print("The actual subject of the email is: \n")
        print(subject_i)
        print()
        print("The predicted subject of the email is (as a bag of words): \n")
        print(predicted_subject)
        print()

main()