# -*- coding: utf-8 -*-
import torch

class Config(object):
    fdata = "data/labeledTrainData.tsv"
    fembed = "embedding/glove.6B.300d.txt"
    num_epochs = 100#迭代次数，利用提前停止机制，所以可以设置的稍微大一点
    embed_size = 300#词嵌入维度即为embed.size(1)
    num_hiddens = 128#LSTM输出维度，单向LSTM的输出维度
    num_layers = 2#LSTM深度，即为LSTM的层数
    bidirectional = True#是否双向LSTM
    batch_size = 64#批量尺寸
    labels = 1#神经网络输出维度，就是分类数，可以设置成one-hot和交叉熵
    lr = 0.1#学习率
    device = torch.device('cuda:0')
    use_gpu = True

