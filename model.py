from modules.attention import img_attention_model, proposal_attention_model
from modules.lstm_model import LSTM_model, encoding_model
from utils import reshape_feature
import torch.nn as nn
import torch
import torch.autograd as autograd
import torch.nn.functional as F

class integrate_model(nn.Module):

    def __init__(self, feature_dim, onehot_dim, embedding_dim, hidden_dim, representation_dim):
        super(integrate_model, self).__init__()

##  left img model
        self.sentence_encoding = encoding_model(onehot_dim, embedding_dim, hidden_dim)   

        self.img_lstm = LSTM_model(feature_dim, embedding_dim, hidden_dim)
        self.img_attention_model = img_attention_model(hidden_dim, feature_dim, embedding_dim)
        self.img_linear_1 = nn.Linear(hidden_dim, representation_dim)
        self.img_linear_2 = nn.Linear(feature_dim, representation_dim)

##  right proposal model         
        self.proposal_lstm = LSTM_model(feature_dim, embedding_dim, hidden_dim)
        self.proposal_attention_model = proposal_attention_model(hidden_dim, feature_dim, embedding_dim)
        self.proposal_linear_1 = nn.Linear(hidden_dim, representation_dim)
        self.proposal_linear_2 = nn.Linear(feature_dim, representation_dim)        

        self.init_weights()


    def init_weights(self):
        initrange = 0.1
        self.sentence_encoding.embedding.weight.data.uniform_(-initrange, initrange)
        self.sentence_encoding.embedding.bias.data.fill_(0)
##  img
        self.img_linear_1.bias.data.fill_(0)
        self.img_linear_1.weight.data.uniform_(-initrange, initrange)
        self.img_linear_2.bias.data.fill_(0)
        self.img_linear_2.weight.data.uniform_(-initrange, initrange)
        self.img_lstm.embedding_1.weight.data.uniform_(-initrange, initrange)
        self.img_lstm.embedding_1.bias.data.fill_(0)
        self.img_lstm.embedding_2.weight.data.uniform_(-initrange, initrange)
        self.img_lstm.embedding_2.bias.data.fill_(0)

## proposal
        self.proposal_linear_1.bias.data.fill_(0)
        self.proposal_linear_1.weight.data.uniform_(-initrange, initrange)
        self.proposal_linear_2.bias.data.fill_(0)
        self.proposal_linear_2.weight.data.uniform_(-initrange, initrange)
        self.proposal_lstm.embedding_1.weight.data.uniform_(-initrange, initrange)
        self.proposal_lstm.embedding_1.bias.data.fill_(0)
        self.proposal_lstm.embedding_2.weight.data.uniform_(-initrange, initrange)
        self.proposal_lstm.embedding_2.bias.data.fill_(0)



    def forward(self, img_feature, proposal_img_feature, questions_feature, questions_mask, proposal_questions_feature, proposal_questions_mask, questions_len, patch_lengths, img_output_container, proposal_output_container, max_questions):


        attend_img_container = autograd.Variable(torch.rand(img_output_container.size(0), 512).float(), requires_grad=True).cuda()
        attend_proposal_container = autograd.Variable(torch.rand(proposal_output_container.size(0), 512).float(), requires_grad=True).cuda()

        for ii in range(max_questions):
##   attention
            attended_img_feature = self.img_attention_model(img_feature, self.img_lstm.hidden) 
            attended_proposal_feature = self.proposal_attention_model(proposal_img_feature, self.proposal_lstm.hidden, patch_lengths)   
##   img_lstm
            img_lstm_output = self.img_lstm(attended_img_feature, questions_feature[:,ii,:], questions_mask[:,ii])
            proposal_lstm_output = self.proposal_lstm(attended_proposal_feature, proposal_questions_feature[:,ii,:], proposal_questions_mask[:,ii])
            img_output_container[:,ii,:] = img_lstm_output
            proposal_output_container[:,ii,:] = proposal_lstm_output

            count = 0
            for jj in range(img_output_container.size(0)):
            	if ii == questions_len[jj]:
            	    attend_img_container[jj,:] = attended_img_feature[jj,:]
                    attend_proposal_container[count:count + patch_lengths[jj], :] = attended_proposal_feature[count:count + patch_lengths[jj], :]
                count = count + patch_lengths[jj]

##   select the last hidden states      
        img_lstm_representation = autograd.Variable(torch.rand(img_output_container.size(0), 1, img_output_container.size(2)).float(), requires_grad=True).cuda()
        proposal_lstm_representation = autograd.Variable(torch.rand(proposal_output_container.size(0), 1, proposal_output_container.size(2)).float(), requires_grad=True).cuda()  
        count = 0      
        for ii in range(img_output_container.size(0)):
            img_lstm_representation[ii,:,:] = img_output_container[ii, questions_len[ii],:]
            proposal_lstm_representation[count:count + patch_lengths[ii],:,:] = proposal_output_container[count:count + patch_lengths[ii], questions_len[ii],:]
            count = count + patch_lengths[ii]

##  image level
        img_lstm_representation = F.tanh(self.img_linear_1(img_lstm_representation.squeeze(1)))
        attend_img_container = F.tanh(self.img_linear_2(attend_img_container.squeeze(1)))
        img_QA_representation = torch.cat([attend_img_container, img_lstm_representation], 1)


        proposal_lstm_representation = F.tanh(self.proposal_linear_1(proposal_lstm_representation.squeeze(1)))
        attend_proposal_container = F.tanh(self.proposal_linear_2(attend_proposal_container.squeeze(1)))
        proposal_QA_representation = torch.cat([attend_proposal_container, proposal_lstm_representation], 1)
        
   
    	return img_QA_representation, proposal_QA_representation


