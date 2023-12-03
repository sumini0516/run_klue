import csv
import pandas
import pandas as pd
import tqdm
import json

f = open('C:/Users/user/Documents/cuknlp/intoCNS/KoNLU_v5.0/dataset_0527_v1/search_disease_0526_v3.csv', 'r',
         encoding='cp949')
rdr = csv.reader(f)
l = []
for line in rdr:
    l.append(line)

final_list1_1 = {"keys": []}
final_list1_2 = {"keys": []}
final_list1_3 = {"keys": []}
final_list2_1 = {"keys": []}
final_list2_2 = {"keys": []}
final_list2_3 = {"keys": []}
list_num_code = []
final_num_list = []
list_num_symptom = ["B-symptom", "I-symptom", "B-animal", "I-animal", "B-name", "I-name", "O"]

# print(l[0])

# num_list
for data in l:
    # print(data[0])
    # print(data[1])
    tag_list = data[1].split()
    for tag in tag_list:
        if tag == "O":
            continue
        else:
            if tag.split("-")[1] not in list_num_code:
                list_num_code.append(tag.split("-")[1])

list_num_code.sort()
for num in list_num_code:
    final_num_list.append("B-" + num)
    final_num_list.append("I-" + num)

final_num_list.append("O")
print(final_num_list)

file_index = 0
# 하하하하하하
for i in range(len(l)):
    dict1 = {}
    dict2 = {}
    list1 = []
    list2 = []
    print("====================")
    print(i + 1)
    print(l[i][0])
    print(l[i][1])
    # token
    list_token = []
    for token in l[i][0]:
        list_token.append(token)
    l[i][0] = l[i][0].split()
    l[i][1] = l[i][1].split()

    # sentence
    str1 = ""
    str2 = ""
    for j in range(len(l[i][0])):
        if l[i][1][j] == "O":
            str1 += l[i][0][j] + " "
            str2 += l[i][0][j] + " "
        elif l[i][1][j].split("-")[0] == "B":
            str1 = str1 + "<" + l[i][0][j] + " "
            str2 = str2 + "<" + l[i][0][j] + " "
        elif l[i][1][j].split("-")[0] == "I":
            str1 = str1 + l[i][0][j] + " "
            str2 = str2 + l[i][0][j] + " "
        elif l[i][1][j].split("-")[0] == "E":
            str1 = str1 + l[i][0][j] + ":" + (l[i][1][j].split("-")[1]) + ">" + " "
            if l[i][1][j].split("-")[1] not in ["animal", "name"]:
                str2 = str2 + l[i][0][j] + ":" + "symptom" + ">" + " "
            else:
                str2 = str2 + l[i][0][j] + ":" + (l[i][1][j].split("-")[1]) + ">" + " "
        elif l[i][1][j].split("-")[0] == "S":
            str1 = str1 + "<" + l[i][0][j] + ":" + (l[i][1][j].split("-")[1]) + ">" + " "
            if l[i][1][j].split("-")[1] not in ["animal", "name"]:
                str2 = str2 + "<" + l[i][0][j] + ":" + "symptom" + ">" + " "
            else:
                str2 = str2 + "<" + l[i][0][j] + ":" + (l[i][1][j].split("-")[1]) + ">" + " "

    # ner_num
    str3 = ""
    str4 = ""
    str5 = ""
    num = 1
    compare = len(l[i][1])
    compare_num = 1
    sentence = l[i][0]
    ners = l[i][1]
    print("sentence:", sentence)
    print("ners:", ners)

    for n in range(len(ners)):
        ner = ners[n]
        key = ner.split("-")
        # print("key:", key)
        if ner == 'O':
            for tagging in range(len(sentence[n])):
                str3 = str3 + str(final_num_list.index("O")) + " "
                str4 = str4 + str(list_num_symptom.index("O")) + " "
                str5 = str5 + str(("O")) + " "
            str3 = str3 + str(final_num_list.index("O")) + " "
            str4 = str4 + str(list_num_symptom.index("O")) + " "
            str5 = str5 + str(("O")) + " "
        elif key[0] == "B":
            for tagging in range(len(sentence[n])):
                if tagging == 0:
                    str3 = str3 + str(final_num_list.index("B-" + key[1])) + " "
                    str5 = str5 + str("B-" + key[1]) + " "
                    if key[1] in ["animal", "name"]:
                        str4 = str4 + str(list_num_symptom.index("B-" + key[1])) + " "
                    else:
                        str4 = str4 + str(list_num_symptom.index("B-symptom")) + " "
                else:
                    str3 = str3 + str(final_num_list.index("I-" + key[1])) + " "
                    str5 = str5 + str("I-" + key[1]) + " "
                    if key[1] in ["animal", "name"]:
                        str4 = str4 + str(list_num_symptom.index("I-" + key[1])) + " "
                    else:
                        str4 = str4 + str(list_num_symptom.index("I-symptom")) + " "
            str3 = str3 + str(final_num_list.index("I-" + key[1])) + " "
            str5 = str5 + str("I-" + key[1]) + " "
            if key[1] in ["animal", "name"]:
                str4 = str4 + str(list_num_symptom.index("I-" + key[1])) + " "
            else:
                str4 = str4 + str(list_num_symptom.index("I-symptom")) + " "

        elif key[0] == "I":
            for tagging in range(len(sentence[n])):
                str3 = str3 + str(final_num_list.index("I-" + key[1])) + " "
                str5 = str5 + str("I-" + key[1]) + " "
                if key[1] in ["animal", "name"]:
                    str4 = str4 + str(list_num_symptom.index("I-" + key[1])) + " "
                else:
                    str4 = str4 + str(list_num_symptom.index("I-symptom")) + " "
            str3 = str3 + str(final_num_list.index("I-" + key[1])) + " "
            str5 = str5 + str("I-" + key[1]) + " "
            if key[1] in ["animal", "name"]:
                str4 = str4 + str(list_num_symptom.index("I-" + key[1])) + " "
            else:
                str4 = str4 + str(list_num_symptom.index("I-symptom")) + " "

        elif key[0] == "E":
            for tagging in range(len(sentence[n])):
                str3 = str3 + str(final_num_list.index("I-" + key[1])) + " "
                str5 = str5 + str("I-" + key[1]) + " "
                if key[1] in ["animal", "name"]:
                    str4 = str4 + str(list_num_symptom.index("I-" + key[1])) + " "
                else:
                    str4 = str4 + str(list_num_symptom.index("I-symptom")) + " "
            str3 = str3 + str(final_num_list.index("O")) + " "
            str5 = str5 + str("O") + " "
            str4 = str4 + str(list_num_symptom.index("O")) + " "

        elif key[0] == "S":
            for tagging in range(len(sentence[n])):
                if tagging == 0:
                    str3 = str3 + str(final_num_list.index("B-" + key[1])) + " "
                    str5 = str5 + str("B-" + key[1]) + " "
                    if key[1] in ["animal", "name"]:
                        str4 = str4 + str(list_num_symptom.index("B-" + key[1])) + " "
                    else:
                        str4 = str4 + str(list_num_symptom.index("B-symptom")) + " "
                else:
                    str3 = str3 + str(final_num_list.index("I-" + key[1])) + " "
                    str5 = str5 + str("I-" + key[1]) + " "
                    if key[1] in ["animal", "name"]:
                        str4 = str4 + str(list_num_symptom.index("I-" + key[1])) + " "
                    else:
                        str4 = str4 + str(list_num_symptom.index("I-symptom")) + " "
            str3 = str3 + str(final_num_list.index("O")) + " "
            str5 = str5 + str("O") + " "
            str4 = str4 + str(list_num_symptom.index("O")) + " "
    num += 1
    final_str3 = []
    final_str4 = []
    li = str3.split()
    li2 = str4.split()
    for s in range(len(li) - 1):
        final_str3.append(int(li[s]))
    for s in range(len(li2) - 1):
        final_str4.append(int(li2[s]))

    dict1['sentence'] = str1
    dict1['tokens'] = list_token
    dict1['ner_tags'] = final_str3

    dict2['sentence'] = str2
    dict2['tokens'] = list_token
    dict2['ner_tags'] = final_str4

    if i < 346:
        final_list1_1["keys"].append(dict1)
        final_list2_1["keys"].append(dict2)
    elif 346 <= i < 386:
        final_list1_2["keys"].append(dict1)
        final_list2_2["keys"].append(dict2)
    else:
        final_list1_3["keys"].append(dict1)
        final_list2_3["keys"].append(dict2)

    print(list_token)
    print("str1:", str1)
    print("str2:", str2)
    print("str3:", str3)
    print("str4:", str4)
    print("str5:", str5)

print("======================================")
print("final_num_list:", len(final_num_list))
with open('./final_code_train.json', 'w', encoding='utf-8') as f:
    json.dump(final_list1_1, f, indent=4)

with open('./final_code_test.json', 'w', encoding='utf-8') as f:
    json.dump(final_list1_2, f, indent=4)

with open('./final_code_val.json', 'w', encoding='utf-8') as f:
    json.dump(final_list1_3, f, indent=4)

with open('./final_symptom_train.json', 'w', encoding='utf-8') as f:
    json.dump(final_list2_1, f, indent=4)

with open('./final_symptom_test.json', 'w', encoding='utf-8') as f:
    json.dump(final_list2_2, f, indent=4)

with open('./final_symptom_val.json', 'w', encoding='utf-8') as f:
    json.dump(final_list2_3, f, indent=4)

f.close()
