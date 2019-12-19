import io
import pickle
import codecs
import sys
import pandas as pd
import numpy as np
from collections import deque  
import pdb
import jieba
import jieba.posseg as pseg

df = open('training_data.pkl','rb')
train=pickle.load(df)
df.close()

relation2id = {}
with codecs.open('relationTemplate.txt','r','utf-8') as input_data:
    for line in input_data.readlines():
        relation2id[line.split()[0]] = int(line.split()[1])
    input_data.close()

id2relation = {}
for key, value in relation2id.items():
    id2relation[value] = key

sentence_data = []
tag = []
entity = []

for item in train:
  for relation_item in item[1]:
    tmp_list = relation_item.split(",")
    sentence_data.append([item[0][0],tmp_list[0],tmp_list[1],tmp_list[2]])
    tag.append(tmp_list[2])
    entity.append([tmp_list[0],tmp_list[1]])

test_sentence_data = sentence_data[-900:]
# sentence_data = sentence_data[:-50]

for pair in entity:
  for word_entity in pair:
    jieba.add_word(word_entity)

import collections
def flatten(x):
    result = []
    for el in x:
        if isinstance(x, collections.Iterable) and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

datas = deque()
labels = deque()
positionE1 = deque()
positionE2 = deque()

max_split = 0
for single_sentence in sentence_data:
    words = pseg.cut(single_sentence[0])
    position1 = []
    position2 = []
    sentence = []
    for word, flag in words:
        sentence.append(word)
    max_split = len(sentence) if len(sentence)>max_split else max_split
    try:
        try:
            index1 = sentence.index(single_sentence[1])
        except:
            try:
                target = single_sentence[1]
                search_list=[i for i,x in enumerate(sentence) if x.find(target)!=-1]
                index1 = search_list[0]
            except:
                try:
                    target = single_sentence[1]
                    search_list=[i for i,x in enumerate(sentence) if x.find(target[:2])!=-1]
                    index1 = search_list[0]
                except:
                    target = single_sentence[1]
                    search_list=[i for i,x in enumerate(sentence) if x.find(target[:1])!=-1]
                    index1 = search_list[0]
        try:
            index2 = sentence.index(single_sentence[2])
        except:
            try:
                target = single_sentence[2]
                search_list=[i for i,x in enumerate(sentence) if x.find(target)!=-1]
                index2 = search_list[0]
            except:
                try:
                    target = single_sentence[2]
                    search_list=[i for i,x in enumerate(sentence) if x.find(target[:2])!=-1]
                    index2 = search_list[0]
                except:
                    target = single_sentence[2]
                    search_list=[i for i,x in enumerate(sentence) if x.find(target[:1])!=-1]
                    index2 = search_list[0]
    except:
        continue

    for i,word in enumerate(sentence):
        position1.append(i-index1)
        position2.append(i-index2)
    datas.append(sentence)
    labels.append(relation2id[single_sentence[3]])
    positionE1.append(position1)
    positionE2.append(position2)

all_words = flatten(datas)
sort_allwords = pd.Series(all_words)

sort_allwords = sort_allwords.value_counts()

set_words = sort_allwords.index

set_ids = range(1, len(set_words)+1)
word2id = pd.Series(set_ids, index=set_words)
id2word = pd.Series(set_words, index=set_ids)

word2id["BLANK"]=len(word2id)+1
word2id["UNKNOW"]=len(word2id)+1
id2word[len(id2word)+1]="BLANK"
id2word[len(id2word)+1]="UNKNOW"
#print "word2id",id2word

max_split=50

def X_padding(words,max_len):
    """把 words 转为 id 形式，并自动补全位 max_len 长度。"""
    ids = []
    for i in words:
        if i in word2id:
            ids.append(word2id[i])
        else:
            ids.append(word2id["UNKNOW"])
    if len(ids) >= max_len: 
        return ids[:max_len]
    ids.extend([word2id["BLANK"]]*(max_len-len(ids))) 

    return ids

def pos(num):
    if num<-40:
        return 0
    if num>=-40 and num<=40:
        return num+40
    if num>40:
        return 80

def position_padding(words,max_len):
    words = [pos(i) for i in words]
    if len(words) >= max_len:  
        return words[:max_len]
    words.extend([81]*(max_len-len(words))) 
    return words

df_data = pd.DataFrame({'words': datas, 'tags': labels,'positionE1':positionE1,'positionE2':positionE2}, index=range(len(datas)))

df_data['words'] = df_data['words'].apply(X_padding,args = (max_split,))
df_data['positionE1'] = df_data['positionE1'].apply(position_padding,args = (max_split,))
df_data['positionE2'] = df_data['positionE2'].apply(position_padding,args = (max_split,))
df_data['tags'] = df_data['tags']

datas = np.asarray(list(df_data['words'].values))
datas[2]
labels = np.asarray(list(df_data['tags'].values))
positionE1 = np.asarray(list(df_data['positionE1'].values))
positionE2 = np.asarray(list(df_data['positionE2'].values))
print(datas.shape)
print(labels.shape)
print(positionE1.shape)
print(positionE2.shape)

# import pickle
# with open('test_trainset.pkl', 'wb') as outp:
# 	pickle.dump(word2id, outp)
# 	pickle.dump(id2word, outp)
# 	pickle.dump(relation2id, outp)
# 	pickle.dump(datas, outp)
# 	pickle.dump(labels, outp)
# 	pickle.dump(positionE1, outp)
# 	pickle.dump(positionE2, outp)
# print('** Finished saving the data.')
