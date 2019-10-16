import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F

class LSTM_model(nn.Module):

## this class implements the language and vision integration LSTM

    def __init__(self, feature_dim, embedding_dim, hidden_dim):
        super(LSTM_model, self).__init__()

        self.lstm = nn.LSTM(2 * embedding_dim, hidden_dim, batch_first=True)
        self.embedding_1 = nn.Linear(hidden_dim, embedding_dim)
        self.embedding_2 = nn.Linear(feature_dim, embedding_dim)
        self.hidden_dim = hidden_dim
        self.hidden = self.init_hidden(1)
        self.drop = nn.Dropout(0.5)

    def init_hidden(self, batch_size):
        return (autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim)).cuda(), 
                autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim)).cuda())


    def forward(self, img_feature, QA_feature, questions_mask):

##  embedding the language sequence
        QA_feature = self.drop(F.tanh(self.embedding_1(QA_feature)))  ## N*512
        questions_mask = autograd.Variable(questions_mask.unsqueeze(1).expand(QA_feature.size()), requires_grad=False).cuda()
        masked_QA_feature = QA_feature * questions_mask
        masked_QA_feature = masked_QA_feature.unsqueeze(1)
##  embedding the img feature
        img_feature = self.drop(F.tanh(self.embedding_2(img_feature).unsqueeze(1)))
        concatenate_feature = torch.cat([masked_QA_feature, img_feature], 2)

        lstm_output, self.hidden = self.lstm(concatenate_feature, self.hidden)

        return lstm_output



class encoding_model(nn.Module):


# this class implements the sentence LSTM
    def __init__(self, onehot_dim, embedding_dim, hidden_dim):  
        super(encoding_model, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.embedding = nn.Linear(onehot_dim, embedding_dim)
        self.hidden = self.init_hidden(1, 1)
        self.drop = nn.Dropout(0.5)

    def init_hidden(self, batch_size, max_questions):
        return (autograd.Variable(torch.zeros(1, max_questions*batch_size, self.hidden_dim)).cuda(),
                autograd.Variable(torch.zeros(1, max_questions*batch_size, self.hidden_dim)).cuda())

    def forward(self, embeds, words_mask, words_len):

        embeds = self.drop(F.tanh(self.embedding(embeds)))
        words_mask = autograd.Variable(words_mask.unsqueeze(2).expand(embeds.size()), requires_grad=False).cuda()
        lstm_output, self.hidden = self.lstm(
            embeds*words_mask, self.hidden)
        output = autograd.Variable(torch.rand(lstm_output.size(0), 1, lstm_output.size(2)).float(), requires_grad=True).cuda()
        for ii in range(lstm_output.size(0)):
            output[ii,:,:] = lstm_output[ii, words_len[ii], :]

        return output
