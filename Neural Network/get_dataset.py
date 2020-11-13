import random
import json
import torch
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torchtext
from torchtext import data
import spacy
import pandas as pd
import numpy as np
import time
import torch.nn.utils.rnn as tnt
import sklearn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
file = "/content/drive/My Drive/Colab Notebooks/Chat bot/intents.json"
path  = "/content/drive/My Drive/Colab Notebooks/Chat bot"

data2 = dict()

def get_dataframe(filename):
    global label
    global tags
    global labeldecode
    with open(file, 'r') as json_data:
        intents = json.load(json_data)

    # Converting to dataframe
    tags = []
    patterns = []
    patterns_full = []
    responses = []
    response_full = []
    for intent in intents['intents']:
        tag = intent['tag']
        # add to tag list
        tags.append(tag)
        for pattern in intent['patterns']:
            patterns.append(pattern)
        patterns_full.append(patterns)
        patterns = []
        for response in intent['responses']:
            responses.append(response)
        response_full.append(responses)
        responses = []


    label = []
    # Label encode the labels
    for i, tag in enumerate(tags):
        label.append(i)
    print(label)
    responses = dict()
    for i in range(len(tags)):
            responses[ int(label[i])] =str(response_full[i])
    print(responses)

    

    labeldecode = dict()
    for i in range(len(label)):
        labeldecode[label[i]] = tags[i]
    print(labeldecode)

    data1 = dict()
    # Dictionary
    print(len(patterns_full[0]))
    for i in range(len(tags)):
        for j in range(len(patterns_full[i])):
            data1[str(patterns_full[i][j])] = int(label[i])

    print(data1)
    df = pd.DataFrame(list(data1.items()),columns = ['text','label'])
    print(df)
    print("=================================")
    df.value_counts('label')
    return df

df = get_dataframe(file)
print(label)
print(labeldecode)

train = df
validation = train
train.to_csv('/content/drive/My Drive/Colab Notebooks/Chat bot/train.tsv', sep='\t', index=False)
validation.to_csv('/content/drive/My Drive/Colab Notebooks/Chat bot/valid.tsv', sep='\t', index=False)



# Tokenizing
TEXT = data.Field(sequential=True, lower=True, tokenize='spacy', include_lengths=True)
LABELS = data.Field(sequential=False, use_vocab=False)
def Tokenize(path):
    global TEXT 
    TEXT = data.Field(sequential=True, lower=True, tokenize='spacy', include_lengths=True)
    global LABELS 
    LABELS = data.Field(sequential=False, use_vocab=False)
    train_data, val_data = data.TabularDataset.splits( path=path, train='train.tsv',
                                                              validation='valid.tsv', format='tsv', skip_header=True,
                                                              fields=[('text', TEXT), ('label', LABELS)])

    train_iter, val_iter = data.BucketIterator.splits((train_data, val_data), 
                                                             batch_sizes=(9, 9), 
                                                             sort_key=lambda x: len(x.text), 
                                                             device=None, sort_within_batch=True, 
                                                             repeat=False)

    return train_iter, val_iter, train_data, val_data

train_iter, val_iter, train_data, val_data = Tokenize(path)

TEXT.build_vocab(train_data,val_data)
TEXT.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100)) 
vocab = TEXT.vocab
print("Shape of Vocab:",TEXT.vocab.vectors.shape)
