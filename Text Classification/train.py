# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 15:31:02 2019

@author: hp
"""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from config import Config
from dataPre import Data
from models import BiLSTM, BiLSTM_Attention_x,BiLSTM_Attention,BiLSTM_FFAttention,BiLSTM_HAttention
import torch.optim as optim
import numpy as np
import time

data = Data(Config.fdata)
embed = data.get_embed(Config.fembed)
train_x,train_y,val_x,val_y = data.produce_data(240)
train_set = TensorDataset(torch.LongTensor(train_x), torch.FloatTensor(train_y).reshape(-1,1))
val_set = TensorDataset(torch.LongTensor(val_x), torch.FloatTensor(val_y).reshape(-1,1))
train_iter = DataLoader(train_set, batch_size=Config.batch_size, shuffle=True)
val_iter = DataLoader(val_set, batch_size=Config.batch_size, shuffle=False)




net = BiLSTM_FFAttention(vocab_size=len(embed), embed_size=Config.embed_size,
                   num_hiddens=Config.num_hiddens, num_layers=Config.num_layers,
                   bidirectional=Config.bidirectional, weight=embed,
                   labels=Config.labels, use_gpu=Config.use_gpu)


net.to(Config.device)
loss_function = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters())

def evaluate(loader):
    LOSS = 0
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    for x,target in loader:
        feature = x.to(Config.device)
        score = net(feature)
        
        LOSS += loss_function(score, target.to(Config.device))
        
        score = torch.sigmoid(score).cpu().numpy()
        pred_choice = np.where(score>=0.5,1,0)
        target = target.numpy().astype('int')
        
        # TP    predict 和 label 同时为1
        TP += ((pred_choice == 1) & (target == 1)).sum()
        # TN    predict 和 label 同时为0
        TN += ((pred_choice == 0) & (target == 0)).sum()
        # FN    predict 0 label 1
        FN += ((pred_choice == 0) & (target == 1)).sum()
        # FP    predict 1 label 0
        FP += ((pred_choice == 1) & (target == 0)).sum()

        P = TP / (TP + FP)
        R = TP / (TP + FN)
        F1 = 2 * R * P / (R + P)
        ACC = (TP + TN) / (TP + TN + FP + FN)
    LOSS = LOSS.cpu().numpy()/len(loader)
        
    return F1, ACC, LOSS

max_e, max_ACC = 0, 0.0
interval = 10
file = 'network.pt'

for epoch in range(Config.num_epochs):
    start = time.time()
    for feature, label in train_iter:  
        net.train()
        net.zero_grad()
        feature = feature.to(Config.device)
        label = label.to(Config.device)
        score = net(feature)
        loss = loss_function(score, label)
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        net.eval()
        train_F1, train_ACC, train_LOSS = evaluate(train_iter)
        val_F1, val_ACC, val_LOSS = evaluate(val_iter)
        
    end = time.time()
    runtime = end - start
    print('epoch: %d, train loss: %.4f, train acc: %.4f, train f1: %.4f, test loss: %.4f, test acc: %.4f, test f1: %.4f, time: %.2f' %
          (epoch, train_LOSS, train_ACC, train_F1, val_LOSS, val_ACC, val_F1, runtime))
    
    if val_ACC > max_ACC:
        torch.save(net, file)
        max_e, max_ACC = epoch, val_ACC
    elif epoch - max_e >= interval:
        print('final epoch: %d, final acc: %.4f,'% (max_e, max_ACC))
        break