import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import reshape_feature, tensor2matrix
from torch.autograd import Variable
import torch.nn.functional as F

class img_attention_model(nn.Module):

    def __init__(self, hidden_dim, feature_dim, embedding_dim):
        super(img_attention_model, self).__init__()
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.linear_1 = nn.Linear(feature_dim, embedding_dim)
        self.linear_2 = nn.Linear(hidden_dim, embedding_dim)
        self.linear_3 = nn.Linear(embedding_dim, 1)
        self.drop = nn.Dropout(0.5)

    def forward(self, x, hidden):
        
        batch_size = x.size()[0]
        x = reshape_feature(x)
        x1 = self.drop(F.tanh(self.linear_1(x)))

        x2 = self.drop(F.tanh(self.linear_2(hidden[0].permute(1,0,2))))
        x3 = x2.expand(batch_size, x1.size(1), x.size(2))

        x4 = torch.sum(x1 * x3, 2)
        x5 = F.softmax(x4)
        ## 
        atten = x5.unsqueeze(2).expand_as(x)
        output = torch.sum(x * atten, 1)
#   sumpooling
        return output



class proposal_attention_model(nn.Module):

    def __init__(self, hidden_dim, feature_dim, embedding_dim):
        super(proposal_attention_model, self).__init__()
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.linear_1 = nn.Linear(feature_dim, embedding_dim)
        self.linear_2 = nn.Linear(hidden_dim, embedding_dim)
        self.linear_3 = nn.Linear(embedding_dim, 1)
        self.drop = nn.Dropout(0.5)

    def forward(self, x, hidden, patch_lens):
        
        batch_size = x.size()[0]
        x = reshape_feature(x)
        x1 = self.drop(F.tanh(self.linear_1(x)))

        x2 = self.drop(F.tanh(self.linear_2(hidden[0].permute(1,0,2))))
        x3 = x2.expand(batch_size, x1.size(1) , x.size(2))

        x4 = torch.mean(torch.sum(x1 * x3, 2), 1).unsqueeze(1)

        ## create variables
        x5 = Variable(torch.rand(x4.size()), requires_grad=True).cuda()
        start = 0
        for length in patch_lens:
            x5[start:start+length,:] = torch.t(F.softmax(torch.t(x4[start:start+length,:])))
            start = start+length
            
        ## 
        atten = x5.unsqueeze(2).expand_as(x)
        output = torch.sum(x * atten, 1)
#   sumpooling
        return output    
