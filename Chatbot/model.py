import torch
import torch.nn as nn
import torch.nn.functional as F
#from get_data import*

class CNN_net(nn.Module):
    def __init__(self, vocab, n_filters, filter_sizes, embedding_dim = 100):
        super(CNN_net, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        # two convolutional layers 5x100
        self.conv1 = nn.Conv2d(1, n_filters, kernel_size=(filter_sizes[0], embedding_dim))
        self.conv2 = nn.Conv2d(1, n_filters, kernel_size=(filter_sizes[1], embedding_dim))
        self.fc1 = nn.Linear(100, 5)

    def forward(self, x, lengths=None):
        embedded = self.embedding(x)
        # switch the shapes
        embedded = embedded.permute(1, 0, 2)
        # Add extra dimension
        embedded = embedded.unsqueeze(dim = 1)
        # operate on same output
        x1 = F.relu(self.conv1(embedded))
        x2 = F.relu(self.conv2(embedded))
        # pool on the length
        self.pool1 = nn.MaxPool2d(kernel_size=(x1.shape[2], 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(x2.shape[2], 1))
        x1 = self.pool1(x1)
        x2 = self.pool2(x2)
        # concatenate
        x = torch.cat((x1, x2), dim=1).squeeze(dim = 2).squeeze(dim = 2)
        #  print(f"the shape of output is {np.shape(x)}")
        x = self.fc1(x)
       # print(f"the shape of output is {np.shape(x)}"
        return x


class RNN_net(nn.Module):
    # We will compare which model is more suitable
    def __init__(self,  vocab, hidden_dim = 100, embedding_dim = 100):
        super(RNN_net, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        # use GRU
        self.rnn = nn.GRU(embedding_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 5)

    def forward(self, x, lengths=None):
        x = self.embedding(x)
        # Fix the hidden state problem
        x = tnt.pack_padded_sequence(x, lengths=lengths)
        y, x = self.rnn(x)
        x = self.fc1(x)
        return x

