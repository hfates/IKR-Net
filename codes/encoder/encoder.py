#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 14:00:57 2021

@author: asfand
"""

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from sklearn.model_selection import train_test_split
seed = 52
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
batch_size = 512
epochs = 10000
learning_rate = 1e-4
gaussian=torch.from_numpy(torch.load('ai.m'))
motion=torch.from_numpy(torch.load('m.m'))
kernels=torch.cat((gaussian,motion),0)

X_train, X_test = train_test_split(kernels ,shuffle=True,test_size=0.05, random_state=seed)
train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=True)

class encoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden1 = nn.Linear(in_features=kwargs["input_shape"], out_features=400)
        self.encoder_hidden2 = nn.Linear(in_features=400, out_features=225)
        self.encoder_hidden3 = nn.Linear(in_features=225, out_features=100)
        self.encoder_hidden4 = nn.Linear(in_features=100, out_features=25)
        self.encoder_output = nn.Linear(in_features=25, out_features=10)

    def forward(self, features):
        out = torch.relu(self.encoder_hidden1(features))
        out = torch.relu(self.encoder_hidden2(out))
        out = torch.relu(self.encoder_hidden3(out))
        out = torch.relu(self.encoder_hidden4(out))
        encoded = torch.sigmoid(self.encoder_output(out))
        return encoded

class decoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.decoder_hidden1 = nn.Linear(in_features=10, out_features=25)
        self.decoder_hidden2 = nn.Linear(in_features=25, out_features=100)
        self.decoder_hidden3 = nn.Linear(in_features=100, out_features=225)
        self.decoder_hidden4 = nn.Linear(in_features=225, out_features=400)
        self.decoder_output = nn.Linear(in_features=400, out_features=441)

    def forward(self, encoded):
        out =  torch.relu(self.decoder_hidden1(encoded))
        out =  torch.relu(self.decoder_hidden2(out))
        out =  torch.relu( self.decoder_hidden3(out))
        out =  torch.relu( self.decoder_hidden4(out))
        reconstructed = torch.sigmoid( self.decoder_output(out))
        return reconstructed
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

enc = encoder(input_shape=441).to(device)
dec = decoder(input_shape=25).to(device)
e_opt = optim.Adam(enc.parameters(), lr=learning_rate)
d_opt = optim.Adam(dec.parameters(), lr=learning_rate)

# criterion = nn.MSELoss()
# for epoch in range(epochs):
#     loss = 0
#     for batch in train_loader:
        
#         batch = batch.view(-1, 441).to(device).float()
#         e_opt.zero_grad()
#         d_opt.zero_grad()
#         encoded = enc(batch)
#         decoded = dec(encoded)
#         train_loss = criterion(decoded, batch)
#         train_loss.backward()
#         e_opt.step()
#         d_opt.step()
#         loss += train_loss.item()
    
#     loss = loss / len(train_loader)
    
#     print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, epochs, loss))
#     if epoch%100==0:
#         torch.save(enc.state_dict(), './model2/encoder/'+str(epoch)+'_'+str(loss)+'_pth')
#         torch.save(dec.state_dict(), './model2/decoder/'+str(epoch)+'_'+str(loss)+'_pth')

#######################
enc.load_state_dict(torch.load('../../checkpoints/model2/9900_encoder_pth'), strict=True)   
dec.load_state_dict(torch.load('../../checkpoints/model2/9900_decoder_pth'), strict=True)   

X_test=X_test.to(device).float()
test_loader = torch.utils.data.DataLoader(X_test, batch_size=10, shuffle=False)

test_examples = None

with torch.no_grad():
    for batch_features in test_loader:
        # batch_features = batch_features
        test_examples = batch_features.view(-1, 441)
        encoded=enc(test_examples)
        reconstruction = dec(encoded)
        
        test_examples=test_examples.detach().cpu()
        reconstruction=reconstruction.detach().cpu()
        with torch.no_grad():
            number = 1
            plt.figure(figsize=(20, 4))
            for index in range(number):
                # display original
                ax = plt.subplot(2, number, index + 1)
                plt.imshow(test_examples[index].numpy().reshape(21, 21))
                # plt.gray()
                plt.title('original')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
        
                # display reconstruction
                ax = plt.subplot(2, number, index + 1 + number)
                plt.imshow(reconstruction[index].numpy().reshape(21, 21))
                # plt.gray()
                plt.title('reconstruction')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            plt.show()