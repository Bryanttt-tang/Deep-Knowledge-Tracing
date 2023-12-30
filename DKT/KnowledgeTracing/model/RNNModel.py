import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

class DKT(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, embed_dim):
        super(DKT, self).__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        # self.device=device
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.rnn = nn.LSTM(embed_dim, hidden_dim, layer_dim, batch_first=True)
        # self.rnn = nn.RNN(embed_dim, hidden_dim, layer_dim, batch_first=True,nonlinearity='tanh')
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.pre= nn.Linear(self.input_dim, self.embed_dim)
        self.pre2= nn.Linear(self.embed_dim, round(self.embed_dim/2))
        # self.pre2= nn.Linear(self.input_dim, self.pre_dim)
        self.relu=nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        embed1=self.pre(x)
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(x.device)
        c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(x.device) # cell state for LSTM
        out,(hn, cn) = self.rnn(embed1, (h0, c0))
        # Add a pre-embedding for one-hod encoding
        
        # embed2=self.pre2(self.relu(embed1))
        
        # h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(x.device) # x.size(0) is batch size
        # out,hn = self.rnn(embed1, h0)        
        res = self.sig(self.fc(out))
        return res
