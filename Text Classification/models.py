# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 bidirectional, weight, labels, use_gpu, **kwargs):
        super(BiLSTM, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.use_gpu = use_gpu
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False
        self.encoder = nn.LSTM(input_size=embed_size, hidden_size=self.num_hiddens, batch_first=True,
                               num_layers=num_layers, bidirectional=self.bidirectional,
                               dropout=0)
        self.decoder = nn.Linear(num_hiddens * 2, labels)
        self.drop = nn.Dropout(0.5)
        
    def forward(self, inputs):
        embeddings = self.embedding(inputs)#[batch_size, seq_len, embedded_size],embedding层的输入只能是整型
        drop_1 = self.drop(embeddings)
        states, (hidden, cell) = self.encoder(drop_1)#[seq_len, batch_size, embedded_size]，可以用batch_first=True，就不用转换
        states = torch.transpose(states, 1, 0)
        drop_2 = self.drop(states[-1].squeeze(0))
        outputs = self.decoder(drop_2)
        return outputs


class BiLSTM_Attention_x(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 bidirectional, weight, labels, use_gpu, **kwargs):
        super(BiLSTM_Attention_x, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.use_gpu = use_gpu
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False
        self.encoder = nn.LSTM(input_size=embed_size, hidden_size=self.num_hiddens, batch_first=True,
                               num_layers=num_layers, bidirectional=self.bidirectional,
                               dropout=0)
        self.decoder = nn.Linear(num_hiddens * 2, labels)
        self.drop = nn.Dropout(0.5)
        
    def attention(self, outputs, hidden):
        hidden = hidden.view(self.num_layers,2,outputs.size(0),self.num_hiddens)
        hidden=hidden[-1].squeeze(0)
        merged_state = torch.cat([s for s in hidden],1)
        merged_state = merged_state.squeeze(0).unsqueeze(2)
        weights = torch.bmm(outputs, merged_state)
        weights = torch.nn.functional.softmax(weights.squeeze(2),dim=1).unsqueeze(2)
        attention_out =  torch.bmm(torch.transpose(outputs, 1, 2), weights).squeeze(2)

        return attention_out
        
    def forward(self, inputs):
        embeddings = self.embedding(inputs)#[batch_size, seq_len, embedded_size],embedding层的输入只能是整型
        drop_1 = self.drop(embeddings)
        states, (hidden, cell) = self.encoder(drop_1)#[seq_len, batch_size, embedded_size]，可以用batch_first=True，就不用转换
        atten_outs = self.attention(states,hidden)
        drop_2 = self.drop(atten_outs)
        outputs = self.decoder(drop_2)
        #states = torch.transpose(states, 1, 0)
        #drop_2 = self.drop(states[-1].squeeze(0))
        #outputs = self.decoder(drop_2)
        return outputs


class BiLSTM_Attention(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, bidirectional, weight, labels, use_gpu, **kwargs):
        super(BiLSTM_Attention, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.use_gpu = use_gpu
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False
        self.encoder = nn.LSTM(input_size = embed_size, hidden_size = num_hiddens, batch_first=True,
                               num_layers = num_layers, bidirectional = bidirectional,dropout=0)
        
        self.decoder = nn.Linear(num_hiddens * 2, labels)
        self.drop = nn.Dropout(0.5)
        self.query = nn.Parameter(torch.FloatTensor(num_hiddens * 2, labels))
        self.reset_parameters()
        
    def reset_parameters(self):
        std = (1 / self.num_hiddens*2) ** 0.5
        nn.init.normal_(self.query, mean=0, std=std)
        
        
    def attention(self, outputs):
        outputs_ac = torch.nn.functional.tanh(outputs)
        query = self.query.unsqueeze(0)
        query = query.expand(outputs_ac.size(0),num_hiddens * 2,1)
        weights = torch.bmm(outputs_ac,query)
        weights = torch.nn.functional.softmax(weights.squeeze(2),dim=1).unsqueeze(2)
        attention_out = torch.bmm(torch.transpose(outputs, 1, 2), weights).squeeze(2)

        return torch.nn.functional.tanh(attention_out)
        
    def forward(self, inputs):
        embeddings = self.embedding(inputs)#[batch_size, seq_len, embedded_size],embedding层的输入只能是整型
        drop_1 = self.drop(embeddings)
        states, (hidden, cell) = self.encoder(drop_1)#[seq_len, batch_size, embedded_size]，可以用batch_first=True，就不用转换
        attn_output = self.attention(states)
        drop_2 = self.drop(attn_output)
        outputs = self.decoder(drop_2)
        return outputs


class BiLSTM_HAttention(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, bidirectional, weight, labels, use_gpu, **kwargs):
        super(BiLSTM_HAttention, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.use_gpu = use_gpu
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False
        self.encoder = nn.LSTM(input_size = embed_size, hidden_size = num_hiddens, batch_first=True,
                               num_layers = num_layers, bidirectional = bidirectional,dropout=0)
        
        self.decoder = nn.Linear(num_hiddens * 2, labels)
        self.drop = nn.Dropout(0.5)
        self.keyW = nn.Parameter(torch.FloatTensor(1,num_hiddens * 2, num_hiddens * 2))
        self.keyb = nn.Parameter(torch.FloatTensor(1,num_hiddens * 2))
        self.query = nn.Parameter(torch.FloatTensor(1,1,num_hiddens * 2))
        self.reset_parameters()
        
    def reset_parameters(self):
        std = (1 / self.num_hiddens*2) ** 0.5
        nn.init.normal_(self.keyW, mean=0, std=std)
        nn.init.normal_(self.keyb, mean=0, std=std)
        nn.init.normal_(self.query, mean=0, std=std)
        
        
    def attention(self, outputs, hidden):
        query = self.query 
        query = query.expand(outputs.size(0),outputs.size(1),self.num_hiddens * 2)
        keyW = self.keyW.expand(outputs.size(0),self.num_hiddens * 2,self.num_hiddens * 2)
        keyb = self.keyb.expand(outputs.size(0),self.num_hiddens * 2).unsqueeze(2)
        hidden = hidden.view(self.num_layers,2,outputs.size(0),self.num_hiddens)
        hidden=hidden[-1].squeeze(0)
        merged_state = torch.cat([s for s in hidden],1)
        merged_state = merged_state.squeeze(0).unsqueeze(2)
        key = torch.nn.functional.tanh(torch.bmm(keyW,merged_state)+keyb)
        weights = torch.bmm(query,key)
        weights = torch.nn.functional.softmax(weights.squeeze(2),dim=1).unsqueeze(2)
        attention_out =  torch.bmm(torch.transpose(outputs, 1, 2), weights).squeeze(2)

        return attention_out
        
    def forward(self, inputs):
        embeddings = self.embedding(inputs)#[batch_size, seq_len, embedded_size],embedding层的输入只能是整型
        drop_1 = self.drop(embeddings)
        states, (hidden, cell) = self.encoder(drop_1)#[seq_len, batch_size, embedded_size]，可以用batch_first=True，就不用转换
        attn_output = self.attention(states,hidden)
        drop_2 = self.drop(attn_output)
        outputs = self.decoder(drop_2)
        return outputs


class BiLSTM_FFAttention(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, bidirectional, weight, labels, use_gpu, **kwargs):
        super(BiLSTM_FFAttention, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.use_gpu = use_gpu
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False
        self.encoder = nn.LSTM(input_size = embed_size, hidden_size = num_hiddens, batch_first=True,
                               num_layers = num_layers, bidirectional = bidirectional,dropout=0)
        
        self.decoder1 = nn.Linear(num_hiddens * 2, num_hiddens)
        self.decoder2 = nn.Linear(num_hiddens, labels)
        self.drop = nn.Dropout(0.5)
        
        self.projection = nn.Sequential(
            nn.Linear(num_hiddens*2, num_hiddens),
            nn.ReLU(True),
            nn.Linear(num_hiddens, 1))
        
        
        
    def ff_attention(self, LSTM_outs):
        energy = self.projection(LSTM_outs)
        weights = F.softmax(energy.squeeze(-1), dim=1).unsqueeze(2)
        attention_outs =  torch.bmm(torch.transpose(LSTM_outs, 1, 2), weights).squeeze(2)
        return attention_outs
        
    def forward(self, inputs):
        embeddings = self.embedding(inputs)#[batch_size, seq_len, embedded_size],embedding层的输入只能是整型
        drop_1 = self.drop(embeddings)
        LSTM_outs, (hidden, cell) = self.encoder(drop_1)#[seq_len, batch_size, embedded_size]，可以用batch_first=True，就不用转换
        attn_output = self.ff_attention(LSTM_outs)
        drop_2 = self.drop(attn_output)
        outputs1 = F.relu(self.decoder1(drop_2))
        outputs2 = self.decoder2(outputs1)
        
        return outputs2