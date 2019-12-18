import jieba.posseg as pseg
import re
import pickle

pre_list = []
base_address = "/Users/kylinchan/Documents/Autumn2019-Git/ChineseNER/data/ie_employee/LabeledData_"
for tag_idx in range(1,6,1):
    actual_address = base_address + str(tag_idx)
    tmp_file = open(actual_address)
    tmp_list = tmp_file.readlines()
    every_sentence = []
    every_tag = []
    enter_flag = 1
    for item in tmp_list:
        if(item!='\n'):
            if(enter_flag):
                every_sentence.append(item)
                enter_flag = 0
            else:
                every_tag.append(item)
        else:
            pre_list.append([every_sentence,every_tag])
            every_sentence = []
            every_tag = []
            enter_flag = 1
output = open('pre_data1.pkl', 'wb')
pickle.dump(pre_list, output, -1)
output.close()

pre_list2 = []
jieba_list = []
p1 = re.compile(r'[{}](.*?)[}]', re.S)
for item_tuple in pre_list:
    tmp_sentence = item_tuple[0][0]
    tmp_tag = item_tuple[1]
    add_list = re.findall(p1, tmp_sentence)
    for item in add_list:
        tmp_replace_aim = "{"+item+"}"
        small_item = item.split('/')
        jieba_list.append(small_item)
        tmp_sentence=tmp_sentence.replace(tmp_replace_aim,small_item[0])
    if(tmp_sentence[-1]=='\n'):
        tmp_sentence=tmp_sentence[:-1]
    transfer_tag = []
    for item in tmp_tag:
        tmp_tag_list=item.split("|")
        tmp_name_list=tmp_tag_list[0].split(',')
        tmp_group_list=tmp_tag_list[1].split(',')
        tmp_name_shortest=0
        tmp_group_shortest=0
        for idx in range(len(tmp_name_list)):
            if(len(tmp_name_list[idx])<len(tmp_name_list[tmp_name_shortest])):
                tmp_name_shortest=idx
        for idx in range(len(tmp_group_list)):
            if(len(tmp_group_list[idx])<len(tmp_group_list[tmp_group_shortest])):
                tmp_group_shortest=idx
        tmp_resemble_tag = tmp_name_list[tmp_name_shortest]+","+tmp_group_list[tmp_group_shortest]+","+"E"
        transfer_tag.append(tmp_resemble_tag)
    pre_list2.append([[tmp_sentence],transfer_tag])
output = open('jieba_data1.pkl', 'wb')
pickle.dump(jieba_list, output, -1)
output.close()
output = open('pre_data2.pkl', 'wb')
pickle.dump(pre_list2, output, -1)
output.close()
print(len(pre_list2))
# print(jieba_list)