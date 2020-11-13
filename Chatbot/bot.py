#python -m spacy download en
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torchtext
from torchtext import data
import spacy
import pandas as pd
from model import*
import random
import json

# Decoding dictionary
path =  "/Users/yvielcastillejos/python_code/Chatbot"
labeldecode = dict()
labeldecode = {0: 'greeting', 1: 'goodbye', 2: 'thanks', 3: 'bad', 4: 'funny'}

# get the responses

with open(f"{path}/intents.json", 'r') as json_data:
	intents = json.load(json_data)

# Converting to dataframe
tags = []
responses = []
response_full = []
for intent in intents['intents']:
	tag = intent['tag']
	# add to tag list
	tags.append(tag)
	for response in intent['responses']:
	    responses.append(response)
	response_full.append(responses)
	responses = []

label = []
# Label encode the labels
for i, tag in enumerate(tags):
	label.append(i)
responses = dict()
for i in range(len(tags)):
	responses[ int(label[i])] =response_full[i]

# print(responses)


# Tokeniize the words
def Tokenize(path):
    global TEXT 
    TEXT = data.Field(sequential=True, lower=True, tokenize='spacy', include_lengths=True)
    global LABELS 
    LABELS = data.Field(sequential=False, use_vocab=False)
    train_data, val_data = data.TabularDataset.splits( path=path, train='train.tsv',
                                                              validation='valid.tsv', format='tsv', skip_header=True,
                                                              fields=[('text', TEXT), ('label', LABELS)])
    return  train_data, val_data
train_data, val_data = Tokenize(path)

# Build vocab
TEXT.build_vocab(train_data,val_data)
TEXT.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100)) 
vocab = TEXT.vocab
#print("Shape of Vocab:",TEXT.vocab.vectors.shape)
print("Done loading vocab")


def loadmodels(TEXT_I, path):
    cnn = CNN_net(TEXT_I.vocab, 50,[1,1])
    #rnn = RNN_net(TEXT_I.vocab)

    cnn = torch.load(f'{path}/model_cnn.pt', map_location=torch.device('cpu'))
    #rnn = torch.load(f'{path}model_rnn.pt')

    cnn.eval()
    #rnn.eval()
    return  cnn #, rnn

def tokenizer(text):
    spacy_en = spacy.load('en')
    return [tok.text for tok in spacy_en(text)]

def convert(ans):
    i, prediction = torch.max(ans,1)
    #print(prediction)
    return labeldecode[prediction.item()]
def get_class(ans):
      i, prediction = torch.max(ans,1)
      #print(prediction)
      return prediction.item()

def Tokenize2(path):
    user_data, useless = data.TabularDataset.splits(path=path, train='user.tsv', validation = 'valid.tsv'
                                                              , format='tsv', skip_header=True,
                                                              fields=[('text', TEXT), ('label', LABELS)])
    user_iter, _ = data.BucketIterator.splits((user_data, val_data), 
                                            batch_sizes=(len(user_data),len(user_data)), 
                                            sort_key=lambda x: len(x.text), 
                                            device=None, sort_within_batch=False, 
                                            repeat=False)
    #print("OK")
    return user_iter



def bot():
    while True:
        df = pd.DataFrame()
        data2 = dict()
        cnn = loadmodels(TEXT, path)
        sentence = input(' Enter a sentence:\n')
        if sentence == "Q":
            break
        #print(sentence)
        to_see = dict()
        label = 0
        text = sentence
        to_see[text] = label
        #print(to_see)
        df2 = pd.DataFrame(list(to_see.items()), columns = ['text', 'label'])
        #print(df2)
        df2.to_csv(f"{path}/user.tsv", sep='\t', index=False)
        example = [sentence]
        user_iter = Tokenize2(path)

        '''
        tokens = tokenizer(sentence)
        token_ints = [vocab.stoi[tok] for tok in tokens]
        token_tensor = torch.LongTensor(token_ints).view(-1, 1)
        lengths = torch.Tensor([len(token_ints)])
        train.to_csv('/content/drive/My Drive/Colab Notebooks/Chat bot/user.tsv', sep='\t', index=False)
        print(token_tensor)
        print(lengths)'''
        
        for i, data in enumerate(user_iter, 0):
            # get the inputs; data is a list of [inputs, labels]
            batch_input, batch_input_length = data.text
            batch_labels = (data.label)
           # print("OK")
            cnn_out = cnn(batch_input, batch_input_length)
           # print(cnn_out[0])
            #rnn_out = rnn(token_tensor, lengths).item()
            cnn_ans = convert(cnn_out)
            cnn_ans_class = get_class(cnn_out)
            response_list = responses[cnn_ans_class]
            # print(response_list[0])
            response =  random.choice(response_list)
            
            #rnn_ans = convert(rnn_out)
	    
            #print(f'Model rnn: {rnn_ans} ({response})')
            print(response)
    return

#bot()
