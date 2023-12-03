import csv
import pandas
import pandas as pd
import tqdm
import json
import random

f = open("C:/Users/user/Documents/cuknlp/intoCNS/KoBERT_KLUE/token_classification/ner_dataset_0802_train.csv", 'r',
         encoding='cp949')
rdr = csv.reader(f)
sentence = []
list_entity = ["B-symptom", "I-symptom", "B-animal", "I-animal", "B-name", "I-name", "B-date", "I-date", "B-time",
               "I-time", "B-disease", "I-disease", "B-location", "I-location", "B-hospital", "I-hospital", "O"]
for line in rdr:
    print(line)
    sentence.append(line[1])

# print("길이", len(sentence))
sentence = sentence[1:]
list_final_entity_recognition = []
# print(sentence)
random.shuffle(sentence)

for sen in sentence:
    list = []
    sentence1 = ""
    ner_num = ""
    token_list = []
    # print("sen:", sen)
    list.append(sen)

    # sentence
    sentence1 = sen
    sentence1 = sentence1.replace("<", "")
    sentence1 = sentence1.replace(":", "")
    sentence1 = sentence1.replace(">", "")
    sentence1 = sentence1.replace("symptom", "")
    sentence1 = sentence1.replace("animal", "")
    sentence1 = sentence1.replace("name", "")
    sentence1 = sentence1.replace("date", "")
    sentence1 = sentence1.replace("time", "")
    sentence1 = sentence1.replace("disease", "")
    sentence1 = sentence1.replace("location", "")
    # print("sentence1:", sentence1)

    # token
    for token in sentence1:
        token_list.append(token)
    # print("token_list:", token_list)

    sen_list = sen.split("<")
    # print("sen_list:", sen_list)
    for component in sen_list:
        # print("component",component)
        component_list = component.split(">")
        # print("component_list", component_list)
        for com in component_list:
            com = com.split(":")
            if len(com) > 1:
                for entity in range(len(com[0])):
                    if entity == 0:
                        try:
                            ner_num += str(list_entity.index("B-" + com[1])) + " "
                        except:
                            print(component)
                    else:
                        ner_num += str(list_entity.index("I-" + com[1])) + " "
            else:
                for entity in range(len(com[0])):
                    ner_num += str(list_entity.index("O")) + " "
    # print("ner_num:", ner_num)
    list.append(token_list)
    final_ner_str = []
    li = ner_num.split()
    for s in range(len(li)):
        final_ner_str.append(int(li[s]))
    list.append(final_ner_str)
    list_final_entity_recognition.append(list)

final_list1_1 = {"keys": []}
for i in range(len(list_final_entity_recognition)):
    dict1 = {}
    dict1['sentence'] = list_final_entity_recognition[i][0]
    dict1['tokens'] = list_final_entity_recognition[i][1]
    dict1['ner_tags'] = list_final_entity_recognition[i][2]
    final_list1_1["keys"].append(dict1)

with open('./final_code_train_v5.json', 'w', encoding='utf-8') as f:
    json.dump(final_list1_1, f, indent=4)
