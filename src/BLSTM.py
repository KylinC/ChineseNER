#coding:utf8
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1)

class BiLSTM_ATT(nn.Module):
    def __init__(self,config,embedding_pre):
        super(BiLSTM_ATT,self).__init__()
        self.batch = config['BATCH']
        
        self.embedding_size = config['EMBEDDING_SIZE']
        self.embedding_dim = config['EMBEDDING_DIM']
        
        self.hidden_dim = config['HIDDEN_DIM']
        self.tag_size = config['TAG_SIZE']
        
        self.pos_size = config['POS_SIZE']
        self.pos_dim = config['POS_DIM']
        
        self.pretrained = config['pretrained']
        if self.pretrained:
            #self.word_embeds.weight.data.copy_(torch.from_numpy(embedding_pre))
            self.word_embeds = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_pre),freeze=False)
        else:
            self.word_embeds = nn.Embedding(self.embedding_size,self.embedding_dim)
        
        self.pos1_embeds = nn.Embedding(self.pos_size,self.pos_dim)
        self.pos2_embeds = nn.Embedding(self.pos_size,self.pos_dim)
        self.relation_embeds = nn.Embedding(self.tag_size,self.hidden_dim)
        
        self.lstm = nn.LSTM(input_size=self.embedding_dim+self.pos_dim*2,hidden_size=self.hidden_dim//2,num_layers=1, bidirectional=True)
        self.hidden2tag = nn.Linear(self.hidden_dim,self.tag_size)
        
        self.dropout_emb=nn.Dropout(p=0.5)
        self.dropout_lstm=nn.Dropout(p=0.5)
        self.dropout_att=nn.Dropout(p=0.5)
        
        self.hidden = self.init_hidden()
        
        self.att_weight = nn.Parameter(torch.randn(self.batch,1,self.hidden_dim))
        self.relation_bias = nn.Parameter(torch.randn(self.batch,self.tag_size,1))
        
    def init_hidden(self):
        return torch.randn(2, self.batch, self.hidden_dim // 2)
        
    def init_hidden_lstm(self):
        return (torch.randn(2, self.batch, self.hidden_dim // 2),
                torch.randn(2, self.batch, self.hidden_dim // 2))
                
    def attention(self,H):
        M = F.tanh(H)
        a = F.softmax(torch.bmm(self.att_weight,M),2)
        a = torch.transpose(a,1,2)
        return torch.bmm(H,a)
        
    
                
    def forward(self,sentence,pos1,pos2):

        self.hidden = self.init_hidden_lstm()

        embeds = torch.cat((self.word_embeds(sentence),self.pos1_embeds(pos1),self.pos2_embeds(pos2)),2)
        
        embeds = torch.transpose(embeds,0,1)

        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        
        lstm_out = torch.transpose(lstm_out,0,1)
        lstm_out = torch.transpose(lstm_out,1,2)
        
        lstm_out = self.dropout_lstm(lstm_out)
        att_out = F.tanh(self.attention(lstm_out))
        #att_out = self.dropout_att(att_out)
        
        relation = torch.tensor([i for i in range(self.tag_size)],dtype = torch.long).repeat(self.batch, 1)

        relation = self.relation_embeds(relation)
        
        res = torch.add(torch.bmm(relation,att_out),self.relation_bias)
        
        res = F.softmax(res,1)

        
        return res.view(self.batch,-1)



import numpy as np
import pickle
import sys
import codecs

with open('trainset.pkl', 'rb') as inp:
    word2id = pickle.load(inp)
    id2word = pickle.load(inp)
    relation2id = pickle.load(inp)
    train = pickle.load(inp)
    labels = pickle.load(inp)
    position1 = pickle.load(inp)
    position2 = pickle.load(inp)

with open('testset.pkl', 'rb') as inp:
    test = pickle.load(inp)
    labels_t = pickle.load(inp)
    position1_t = pickle.load(inp)
    position2_t = pickle.load(inp)

import torch
import torch.nn as nn
import torch.optim as optim

import torch.utils.data as D
from torch.autograd import Variable

EMBEDDING_SIZE = len(word2id)+1        
EMBEDDING_DIM = 100

POS_SIZE = 82  #不同数据集这里可能会报错。
POS_DIM = 25

HIDDEN_DIM = 200

TAG_SIZE = len(relation2id)

BATCH = 4
EPOCHS = 100

config={}
config['EMBEDDING_SIZE'] = EMBEDDING_SIZE
config['EMBEDDING_DIM'] = EMBEDDING_DIM
config['POS_SIZE'] = POS_SIZE
config['POS_DIM'] = POS_DIM
config['HIDDEN_DIM'] = HIDDEN_DIM
config['TAG_SIZE'] = TAG_SIZE
config['BATCH'] = BATCH
config["pretrained"]=False

learning_rate = 0.0005

embedding_pre = []
if len(sys.argv)==2 and sys.argv[1]=="pretrained":
    print("use pretrained embedding")
    config["pretrained"]=True
    word2vec = {}
    with codecs.open('vec.txt','r','utf-8') as input_data:   
        for line in input_data.readlines():
            word2vec[line.split()[0]] = map(eval,line.split()[1:])

    unknow_pre = []
    unknow_pre.extend([1]*100)
    embedding_pre.append(unknow_pre) #wordvec id 0
    for word in word2id:
        if word2vec.has_key(word):
            embedding_pre.append(word2vec[word])
        else:
            embedding_pre.append(unknow_pre)

    embedding_pre = np.asarray(embedding_pre)
    print(embedding_pre.shape)

model = BiLSTM_ATT(config,embedding_pre)
#model = torch.load('model/model_epoch20.pkl')
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss(size_average=True)

train = torch.LongTensor(train[:len(train)-len(train)%BATCH])
position1 = torch.LongTensor(position1[:len(train)-len(train)%BATCH])
position2 = torch.LongTensor(position2[:len(train)-len(train)%BATCH])
labels = torch.LongTensor(labels[:len(train)-len(train)%BATCH])
train_datasets = D.TensorDataset(train,position1,position2,labels)
train_dataloader = D.DataLoader(train_datasets,BATCH,True,num_workers=2)

test = torch.LongTensor(test[:len(test)-len(test)%BATCH])
position1_t = torch.LongTensor(position1_t[:len(test)-len(test)%BATCH])
position2_t = torch.LongTensor(position2_t[:len(test)-len(test)%BATCH])
labels_t = torch.LongTensor(labels_t[:len(test)-len(test)%BATCH])
test_datasets = D.TensorDataset(test,position1_t,position2_t,labels_t)
test_dataloader = D.DataLoader(test_datasets,BATCH,True,num_workers=2)


print("train strat")

for epoch in range(EPOCHS):
    print("epoch:",epoch)
    count_right=0
    acc=0
    total=0
    
    for sentence,pos1,pos2,tag in train_dataloader:
        sentence = Variable(sentence)
        pos1 = Variable(pos1)
        pos2 = Variable(pos2)
        y = model(sentence,pos1,pos2)  
        tags = Variable(tag)
        loss = criterion(y, tags)      
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
       
        y = np.argmax(y.data.numpy(),axis=1)

        for y1,y2 in zip(y,tag):
            if y1==y2:
                acc+=1
            total+=1

    for sentence,pos1,pos2,tag in test_dataloader:
        sentence = Variable(sentence)
        pos1 = Variable(pos1)
        pos2 = Variable(pos2)
        y = model(sentence,pos1,pos2)
        y = np.argmax(y.data.numpy(),axis=1)
        for y1,y2 in zip(y,tag):
            if y1==y2:
                count_right+=1
        
    print("train:",100*float(acc)/total,"%")
    print("test:",100*float(count_right)/900,"%")
    