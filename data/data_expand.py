import jieba
import jieba.posseg as pseg
import re
import pickle

df = open('pre_data2.pkl','rb')
pre_data=pickle.load(df)
df.close()

# pickle add expanded_dict
ex_dict_df = open('jieba_data1.pkl','rb')
ex_dict = pickle.load(ex_dict_df)
ex_dict_df.close()
for item in ex_dict:
    jieba.add_word(item[0], freq=None, tag=item[1])

# print(pre_data)
# cut and expand the data
for big_tuple in pre_data:
    tmp_sentence = big_tuple[0][0]
    aim_nr = []
    aim_nt = []
    words = pseg.cut(tmp_sentence) 

    for w in words:
        if(w.flag=='nr'):
            aim_nr.append(w.word)
        if(w.flag=='nt'):
            aim_nt.append(w.word)
    for item_nr in aim_nr:
        for item_nt in aim_nt:
            compare_sentence = item_nr+","+item_nt+",E"
            negtive_sentence = item_nr+","+item_nt+",N"
            if compare_sentence in big_tuple[1]:
                pass
            else:
                big_tuple[1].append(negtive_sentence)
output = open('pre_data3.pkl', 'wb')
pickle.dump(pre_data, output, -1)
output.close()
# print(pre_data)