import numpy as np
import os
import glob
from random import shuffle
import math
import h5py
import torch
import torch.autograd as autograd
import json
import torch.nn as nn
import time
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.tokenize import TweetTokenizer

tknzr = TweetTokenizer(preserve_case=False)


def reshape_feature(features):

    B, C, H, W = list(features.size())
    features = features.view(B, C, W * H)
    features = features.transpose(1, 2)  # Bx(W*H)xC
    return features


def tensor2matrix(x):

    csize = x.size(2)
    x = x.contiguous().view(-1, csize)
    return x 


def init_hidden(proposal_num, hidden_dim):   ## check the dimention 

    return Variable(torch.zeros(1, proposal_num, hidden_dim)).cuda()


def proc_configs(config):
    if not os.path.exists(config['weights_dir']):
        os.makedirs(config['weights_dir'])
        print "Creat folder: " + config['weights_dir']

    return config


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def unpack_configs(config):
    train_info = json.load(open(config['train_info'], 'r'))
    train_info = train_info['summary']
    val_info = json.load(open(config['val_info'], 'r'))
    val_info = val_info['summary']
    test_info = json.load(open(config['test_info'], 'r'))
    test_info = test_info['summary']
    return train_info, val_info, test_info


def pad(tensor, length):
#    tensor = Variable.data
    if tensor.size(1) < length:
        padded = torch.cat([tensor, torch.zeros(tensor.size(0), length - tensor.size(1), *tensor.size()[2:])], 1)
    else:
        padded = tensor
    return padded

def detach(states):
    return [state.detach() for state in states] 


def sen_to_fea(data_source, seq_idxes, word_dict):

## pad words and sentences

    QA_container = []
    questions_num = []
    words_num = []

    for batch_idx in seq_idxes:
        questions = data_source[batch_idx]['question']
        questions_num.append(len(questions))
    max_questions = max(questions_num)


    for batch_idx in seq_idxes:
        questions = data_source[batch_idx]['question']
        for idx in range(max_questions):
            if idx < len(questions):
                words = tknzr.tokenize(questions[idx])
                words.remove('?')
                words_num.append(len(words))
            else:
                words_num.append(0)

    max_words = max(words_num) + 1
    words_mask = torch.zeros(max_questions*len(seq_idxes), max_words)
    questions_mask = torch.zeros(len(seq_idxes), max_questions)

#  
    count = 0
    count_2 = 0
    QA_feature = torch.zeros(max_questions*len(seq_idxes), max_words, 4896)
    for batch_idx in range(len(seq_idxes)):
        questions = data_source[seq_idxes[batch_idx]]['question']
        answers = data_source[seq_idxes[batch_idx]]['ans']
 # iterate over questions
        for idx in range(max_questions):
            sen_feature = torch.zeros(max_words, 4896)
            if idx < questions_num[batch_idx]:
                words = tknzr.tokenize(questions[idx]) 
                words.remove('?')              
                words.append(answers[idx])
 # iterate over words
                for sub_idx in range(max_words):
                    if sub_idx < len(words):                    
                        if words[sub_idx] in word_dict.keys():
                            valid_idx = word_dict[words[sub_idx]]
                            sen_feature[sub_idx, valid_idx] = 1.0
                words_mask[count, 0:len(words)] = 1.0
            count = count + 1
            
            QA_feature[count_2,:,:] = sen_feature
            count_2 = count_2 + 1

        questions_mask[batch_idx, 0:len(questions)] = 1.0 


    return QA_feature, questions_mask, words_mask, words_num, questions_num, max_questions



def get_batch(data_source, seq_idxes, word_dict, patch_lengths, config, flag=True):


    seq_labels = np.zeros((len(seq_idxes),), 'float32')
    seq_qua, questions_mask, words_mask, words_num, questions_num, max_questions = sen_to_fea(data_source, seq_idxes, word_dict)


    sign = 0        
    for idx in range(len(seq_idxes)):

        seq_labels[idx,] = data_source[seq_idxes[idx]]['gt_label']
        if idx > 0:
            seq_labels[idx,] = seq_labels[idx,] + sign
        sign = sign + patch_lengths[idx]
    
    seq_labels = torch.from_numpy(seq_labels).type(torch.LongTensor)



    return  seq_labels, seq_qua, questions_mask, words_mask, words_num, questions_num, max_questions


def get_region_batch(data_source, seq_idxes, patch_features, category_list):


    patch_lengths = []


    seq_language_features = np.zeros((0, 80), 'float32')
    seq_spatial = np.zeros((0, 8), 'float32')

    for idx in seq_idxes:
        category = data_source[idx]['category']
        patch_lengths.append(len(category))

    num_patches = sum(patch_lengths)
    seq_features = torch.zeros(num_patches, 512, 7, 7).float()
    start_idx = 0

    for idx in range(len(seq_idxes)):

        spatial = data_source[seq_idxes[idx]]['spatial']
        category = data_source[seq_idxes[idx]]['category']
        temp_feature_container = np.zeros((len(category), 80))
        temp_spatial_container = np.zeros((len(category), 8))
        count = 0
        for (aa, bb) in zip(spatial, category):
            onehot_idx = category_list.index(bb)
            temp_spatial_container[count, :] = aa 
            temp_feature_container[count, onehot_idx] = 1.0 
            count = count + 1
        seq_language_features = np.concatenate((seq_language_features, temp_feature_container))
        seq_spatial = np.concatenate((seq_spatial, temp_spatial_container))

        seq_features[start_idx:start_idx + patch_lengths[idx],:,:,:] = patch_features[idx, 0:patch_lengths[idx],:,:,:] / 10.0
        start_idx = start_idx + patch_lengths[idx]

    seq_language_features = torch.from_numpy(seq_language_features).float()
    seq_spatial = torch.from_numpy(seq_spatial).float()

    return  seq_features, seq_language_features, seq_spatial, patch_lengths
    
  


def dataparallel(model, ngpus, gpu0=0):
    if ngpus==0:
        assert False, "only support gpu mode"
    gpu_list = list(range(gpu0, gpu0+ngpus))
    assert torch.cuda.device_count() >= gpu0 + ngpus
    if ngpus > 1:
        if not isinstance(model, nn.DataParallel):
	    model = nn.DataParallel(model, gpu_list).cuda()
	else:
	    model = model.cuda()
    return model


def encode_sentence(model, questions, words_num, words_mask):
##   sentence encoding
    sentence_embedding = model.sentence_encoding(questions, words_mask, words_num)                

    return sentence_embedding



def train(model, optimizer, criterion, data_source, word_dict, category_list, epoch, config, train_loader):


    train_num = len(data_source)
    n_train_batches = int(math.floor(train_num / config['batch_size']))
    minibatch_range = range(n_train_batches)
    shuffle(minibatch_range)
    model.train()


    correct = 0
    count = 0
    
    for batch_idx, (img_feature, proposal_feature_container, seq_idxes) in enumerate(train_loader):  
 

        proposal_img_feature, proposal_language_feature, proposal_spatial, patch_lengths = get_region_batch(data_source, seq_idxes, proposal_feature_container, category_list)
        targets, questions, questions_mask, words_mask, words_num, questions_num, max_questions = get_batch(data_source, seq_idxes, word_dict, patch_lengths, config)

        img_feature = Variable(img_feature / 10.0).cuda()
        questions = Variable(questions).cuda()        
        proposal_img_feature = Variable(proposal_img_feature).cuda()
        proposal_language_feature = Variable(proposal_language_feature).cuda()
        proposal_spatial = Variable(proposal_spatial).cuda()
        targets = Variable(targets).cuda()

        optimizer.zero_grad()     
        questions_num = [x-1 for x in questions_num]
 
    ## initialize states
        model.sentence_encoding.hidden = model.sentence_encoding.init_hidden(seq_idxes.size(0), max_questions)
        model.img_lstm.hidden = model.img_lstm.init_hidden(seq_idxes.size(0))
        model.proposal_lstm.hidden = model.proposal_lstm.init_hidden(sum(patch_lengths))  


    ## encode the sentence features first
        sentence_feature = encode_sentence(model, questions, words_num, words_mask)
        sentence_feature = sentence_feature.view(seq_idxes.size(0), max_questions, -1)

##      expand the feature for proposal
        proposal_sentence_feature = Variable(torch.rand(sum(patch_lengths), sentence_feature.size(1), sentence_feature.size(2)), requires_grad=True).cuda()
        proposal_questions_mask = torch.zeros(sum(patch_lengths), questions_mask.size(1)).cuda()
        start = 0
        for ii in range(len(patch_lengths)):
            proposal_sentence_feature[start:start+patch_lengths[ii],:,:] = sentence_feature[ii,:,:].expand(patch_lengths[ii], sentence_feature.size(1), sentence_feature.size(2))
            proposal_questions_mask[start:start+patch_lengths[ii],:] = questions_mask[ii,:].expand(patch_lengths[ii], questions_mask.size(1))
            start = start + patch_lengths[ii]

        img_output_container = Variable(torch.rand(seq_idxes.size(0), max_questions, config['hidden_dim']).float(), requires_grad=True).cuda()
        proposal_output_container = Variable(torch.rand(proposal_img_feature.size(0), max_questions, config['hidden_dim']).float(), requires_grad=True).cuda()


        img_prob, proposal_prob = model(img_feature, proposal_img_feature, sentence_feature, questions_mask, proposal_sentence_feature, proposal_questions_mask, questions_num, patch_lengths, img_output_container, proposal_output_container, max_questions)

##  inner product to calculate similarity
        prob = torch.mm(img_prob, torch.t(proposal_prob))

##  generate class mask
        sign_left = 0
        sign_right = 0
        for ii in range(targets.size(0)):
            sign_right = sign_left + patch_lengths[ii]
            if sign_left > 0:
                prob[ii, 0:sign_left] = -100
            if sign_right < prob.size(1):
                prob[ii, sign_right:] = -100
            sign_left = sign_left + patch_lengths[ii]

        loss = criterion(prob, targets)
        loss.backward()
## debug 
#        aa = []
#        for f in model.parameters():
#            aa.append(f.grad)   
        optimizer.step()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.sentence_encoding.parameters(), 0.25)
        torch.nn.utils.clip_grad_norm(model.img_lstm.parameters(), 0.25)
        torch.nn.utils.clip_grad_norm(model.proposal_lstm.parameters(), 0.25)

        if count % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, count * config['batch_size'], train_num,
            100. * count / n_train_batches, loss.data[0])) 
        count = count + 1           


def evaluate(model, data_source, category_list, word_dict, config, test_loader):

    test_num = len(data_source)
    n_test_batches = int(math.floor(test_num / config['test_batch_size']))
    total = 0
    correct = 0
    model.eval()

    for batch_idx, (img_feature, proposal_feature_container, seq_idxes) in enumerate(test_loader):

        proposal_img_feature, proposal_language_feature, proposal_spatial, patch_lengths = get_region_batch(data_source, seq_idxes, proposal_feature_container, category_list)
        targets, questions, questions_mask, words_mask, words_num, questions_num, max_questions = get_batch(data_source, seq_idxes, word_dict, patch_lengths, config, flag=False)

        img_feature = Variable(img_feature / 10.0).cuda()
        questions = Variable(questions).cuda()        
        proposal_img_feature = Variable(proposal_img_feature).cuda()
        proposal_language_feature = Variable(proposal_language_feature).cuda()
        proposal_spatial = Variable(proposal_spatial).cuda()
        targets = Variable(targets).cuda()

    ## initialize states
        model.sentence_encoding.hidden = model.sentence_encoding.init_hidden(seq_idxes.size(0), max_questions)  
        model.img_lstm.hidden = model.img_lstm.init_hidden(seq_idxes.size(0))
        model.proposal_lstm.hidden = model.proposal_lstm.init_hidden(sum(patch_lengths))  
        
        questions_num = [x-1 for x in questions_num]

    ## encode the sentence features first
        sentence_feature = encode_sentence(model, questions, words_num, words_mask)
        sentence_feature = sentence_feature.view(seq_idxes.size(0), max_questions, -1)

##      expand the feature for proposal
        proposal_sentence_feature = Variable(torch.rand(sum(patch_lengths), sentence_feature.size(1), sentence_feature.size(2)), requires_grad=True).cuda()
        proposal_questions_mask = torch.zeros(sum(patch_lengths), questions_mask.size(1)).cuda()
        start = 0
        for ii in range(len(patch_lengths)):
            proposal_sentence_feature[start:start+patch_lengths[ii],:,:] = sentence_feature[ii,:,:].expand(patch_lengths[ii], sentence_feature.size(1), sentence_feature.size(2))
            proposal_questions_mask[start:start+patch_lengths[ii],:] = questions_mask[ii,:].expand(patch_lengths[ii], questions_mask.size(1))
            start = start + patch_lengths[ii]

        img_output_container = Variable(torch.rand(seq_idxes.size(0), max_questions, config['hidden_dim']).float(), requires_grad=True).cuda()
        proposal_output_container = Variable(torch.rand(proposal_img_feature.size(0), max_questions, config['hidden_dim']).float(), requires_grad=True).cuda()


        img_prob, proposal_prob = model(img_feature, proposal_img_feature, sentence_feature, questions_mask, proposal_sentence_feature, proposal_questions_mask, questions_num, patch_lengths, img_output_container, proposal_output_container, max_questions)
##  inner product to calculate similarity       
        prob = torch.mm(img_prob, torch.t(proposal_prob))

        sign_left = 0
        sign_right = 0
        for ii in range(targets.size(0)):
            sign_right = sign_left + patch_lengths[ii]
            if sign_left > 0:
                prob[ii, 0:sign_left] = -100
            if sign_right < prob.size(1):
                prob[ii, sign_right:] = -100
            sign_left = sign_left + patch_lengths[ii]
            

        pred = prob.data.max(1)[1] # get the index of the max log-probability
        total += seq_idxes.size(0)
        correct += pred.eq(targets.data).cpu().sum()

        if batch_idx % 100 == 0:
            print('[{}/{} ({:.0f}%)]'.format(
            batch_idx * config['test_batch_size'], test_num,
            100. * batch_idx / n_test_batches))

    print('Test Accuracy of the model: %f %%' % (100.0 * correct / total))
    return 100.0 * correct / total







