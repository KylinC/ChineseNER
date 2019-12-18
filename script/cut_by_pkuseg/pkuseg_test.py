import pkuseg
import pickle
import jieba
import jieba.posseg as pseg

with open('/Users/kylinchan/Documents/Autumn2019-Git/ChineseNER/data/pre_processed_data/training_data.pkl', 'rb') as inp:
  train = pickle.load(inp)

real_abstract = []
for poplist in train:
  self_list = []
  for raw_item in poplist[1]:
    if(raw_item[-1]=='E'):
      split_item = raw_item.split(',')
      self_list.append((split_item[0],'nr'))
      self_list.append((split_item[1],'nt'))
  real_abstract.append(self_list)

lexicon = []

for tmp_list in real_abstract:
  for tmp_tuple in tmp_list:
    jieba.add_word(tmp_tuple[0],tag=tmp_tuple[1])
    lexicon.append(tmp_tuple[0])

seg = pkuseg.pkuseg(model_name='/Users/kylinchan/Documents/Autumn2019-Git/ChineseNER/script/cut_by_pkuseg/ctb8',user_dict=lexicon) 
text = seg.cut('于永波等在同应邀参加中宣部召开的全国先进典型座谈会的军队代表徐洪刚、韩素云、李国安、邹延龄、第四军医大学学员二大队代表李尔青以及武警部队国旗护卫队代表王建华座谈时，称赞他们的先进事迹是中华民族传统美德和我党我军优良传统的完美结合，体现了我党我军全心全意为人民服务的宗旨，体现了与社会主义市场经济相适应的时代精神。')   
print(text)

words = pseg.cut("于永波等在同应邀参加中宣部召开的全国先进典型座谈会的军队代表徐洪刚、韩素云、李国安、邹延龄、第四军医大学学员二大队代表李尔青以及武警部队国旗护卫队代表王建华座谈时，称赞他们的先进事迹是中华民族传统美德和我党我军优良传统的完美结合，体现了我党我军全心全意为人民服务的宗旨，体现了与社会主义市场经济相适应的时代精神。")
words_seq = []
for word, flag in words:
    words_seq.append(word)
print(words_seq)