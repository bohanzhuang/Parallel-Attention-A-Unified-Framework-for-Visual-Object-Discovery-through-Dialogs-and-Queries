import torch
import torch.autograd as autograd
from torch.autograd import Variable
from torchvision import transforms, datasets
import torch.nn as nn
from read_data import MyDataset
from model import integrate_model
from utils import proc_configs, unpack_configs, repackage_hidden, train, evaluate, init_hidden
import numpy as np
from random import shuffle
import torch.nn.functional as F
import yaml
import math
from random import shuffle
import pickle



with open('./config.yaml', 'r') as f:
    config = yaml.load(f)

config = proc_configs(config)

# global variables

train_info, val_info, test_info = unpack_configs(config)
train_num = len(train_info)
index_shuf = range(train_num)
shuffle(index_shuf)

batch_size = config['batch_size']
test_batch_size = config['test_batch_size']

epochs = config['n_epochs']
learning_rate = config['learning_rate']



# initialize model
embedding_dim = 512
hidden_dim = 512
onehot_dim = 4896
representation_dim = 512
feature_dim = 512
epoch = 0

word_dict = pickle.load(open('word_dict.pkl', 'r'))
category_list = pickle.load(open('category_dict.pkl', 'r'))


train_dataset = MyDataset(train_info, config['feature_dir'], config['train_region_features'])

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True, num_workers=5)


test_dataset = MyDataset(val_info, config['feature_dir'], config['val_region_features'])
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size, 
                                           shuffle=False, num_workers=5)



#----------- training----------
# initialize 

model = integrate_model(feature_dim, onehot_dim, embedding_dim, hidden_dim, representation_dim)
model.cuda()


criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



val_record = []
best_val_accuracy = 0.0
save_frequency = 1


while epoch < epochs:

    epoch = epoch + 1
# resume training    
    if config['resume_train'] and epoch == 1:
        load_epoch = config['load_epoch']
        epoch = load_epoch + 1
        resume_dict = torch.load('./weights/model_' + str(load_epoch) + '.pt')     

        model.load_state_dict(resume_dict)
        learning_rate = np.load(
                config['weights_dir'] + 'lr_' + str(load_epoch) + '.npy')
        val_record = list(
                np.load(config['weights_dir'] + 'val_record.npy'))
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train(model, optimizer, criterion, train_info, word_dict, category_list, epoch, config, train_loader)

#  adjust learning rate
    if epoch % 15 == 0:
        learning_rate /= 10.0

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("Learning_rate = "+str(learning_rate))

# save model 
    if (epoch+1) % save_frequency == 0:
        model_file = config['weights_dir'] + 'model_' + str(epoch) + '.pt'
        torch.save(model.state_dict(), model_file)
        np.save(config['weights_dir'] + 'lr_' + str(epoch) + '.npy',
                       learning_rate) 


    val_accuracy = evaluate(model, val_info, category_list, word_dict, config, test_loader)
    val_record.append([val_accuracy])
    np.save(config['weights_dir'] + 'val_record.npy', val_record)











    












    
  
