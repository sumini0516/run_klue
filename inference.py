from datasets import load_dataset
from transformers import DataCollatorWithPadding, AutoModelForTokenClassification, AutoModelForSequenceClassification, AutoTokenizer, logging, DataCollatorForTokenClassification
from torch.utils.data import DataLoader
import pandas as pd
from eval_metric import metrics
import torch

logging.set_verbosity_error()
raw_datasets1 = {}
label_list1 = ["B-symptom", "I-symptom", "B-animal", "I-animal", "B-name", "I-name", "B-date", "I-date", "B-time",
               "I-time", "B-disease", "I-disease", "B-location", "I-location", "O"]


# raw_datasets2 = {}


def make_labels(example):
    # if self.tokenizer_type == "xlm-sp":

    #     strip_char = "▁"
    # elif self.tokenizer_type == "bert-wp":
    #     strip_char = "##"
    strip_char = "##"

    original_clean_tokens = []
    original_clean_labels = []


    sentence = ""
    for token, tag in zip(example['tokens'], example['ner_tags']):
        sentence += token
        if token == " ":
            continue
        original_clean_tokens.append(token)
        original_clean_labels.append(tag)

    sent_words = sentence.split(" ")
    modi_labels = []
    # modi_labels.append(12)
    char_idx = 0

    for word in sent_words:
        # 안녕, 하세요
        correct_syllable_num = len(word)
        tokenized_word = tokenizer.tokenize(word)
        print("tokenized_word:", tokenized_word)
        # case1: 음절 tokenizer --> [안, ##녕]
        # case2: wp tokenizer --> [안녕]
        # case3: 음절, wp tokenizer에서 unk --> [unk]
        # unk규칙 --> 어절이 통채로 unk로 변환, 단, 기호는 분리
        contain_unk = True if tokenizer.unk_token in tokenized_word else False
        for i, token in enumerate(tokenized_word):
            token = token.replace(strip_char, "")
            if not token:
                modi_labels.append(12)
                continue
            modi_labels.append(original_clean_labels[char_idx])
            if not contain_unk:
                char_idx += len(token)
        if contain_unk:
            char_idx += correct_syllable_num
    # modi_labels.append(12)
    # print(sentence, modi_labels)
    example['sentence'] = sentence
    example['labels'] = modi_labels
    return example


def make_padding_label(example):
    l = len(example["input_ids"])
    labels = example['labels']
    labels = [-100] + labels + ([-100] * (l - len(labels) - 1))
    example['labels'] = labels
    return example


def tokenize_function1(example):
    return tokenizer(example["sentence"], truncation=True, padding='max_length', max_length=64)


def tokenize_function2(example):
    label_to_id = {v: i for i, v in enumerate(label_list2)}
    result = tokenizer(example["sentence1"], truncation=True)
    result["labels"] = 6
    return result


def make_padding_label(example):
    l = len(example["input_ids"])
    labels = example['labels']
    labels = [-100] + labels + ([-100] * (l - len(labels) - 1))
    example['labels'] = labels

    return example


def preprocess1(input):
    # token
    list_token = []
    list_ner_tags = []
    for token in input:
        list_token.append(token)
    # sentence는 건들게 없음
    # ner tags
    for token in list_token:
        list_ner_tags.append(int(14))
    raw_datasets1["sentence"] = input
    raw_datasets1["tokens"] = list_token
    raw_datasets1["ner_tags"] = list_ner_tags
    return raw_datasets1


PATH1 = "C:/Users/user/Documents/cuknlp/intoCNS/KoNLU_v5.0/model_save_final/model_v2.pth"
PATH2 = "C:/Users/user/Documents/cuknlp/intoCNS/KoBERT_KLUE/model_save/model.pth"

# 입력
x = str(input("입력하세요:"))
raw_datasets1 = preprocess1(x)
# raw_datasets2 = preprocess2(x)
print("raw_datasets1:")
print(raw_datasets1)
# print("raw_datasets1:")
# print(raw_datasets2)

pretrained_model = "klue/roberta-large"

# entity
device = torch.device('cuda')
model1 = AutoModelForTokenClassification.from_pretrained(pretrained_model, num_labels=15).to(device)
model1.load_state_dict(torch.load(PATH1))
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
label = make_labels(raw_datasets1)
batch1 = tokenize_function1(raw_datasets1)
# print(batch)
batch1.update(label)

batch1 = make_padding_label(batch1)
batch1.pop('ner_tags')
batch1.pop('sentence')
batch1.pop('tokens')
batch1 = {k: torch.tensor(v).to(device).reshape([1, -1]) for k, v in batch1.items()}
# print(batch2['input_ids'].size())
# print(batch2['token_type_ids'].size())
# print(batch2['attention_mask'].size())
# print(batch2['labels'].size())
# print(batch2)
preds = []
probs = []
output = model1(**batch1)
logits = output.logits
prob = torch.softmax(logits, dim=-1)
predictions = torch.argmax(logits, dim=-1)
print(predictions)
preds.append(predictions)
probs.append(prob)

# intent classification
model2 = AutoModelForSequenceClassification.from_pretrained(pretrained_model, num_labels=7).to(device)
model2.load_state_dict(torch.load(PATH2))
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
raw_datasets = "C:/Users/user/Documents/cuknlp/intoCNS/KoBERT_KLUE/intocns_train.csv"
raw_datasets = load_dataset("csv", data_files=raw_datasets)
example = {'label': 6, 'sentence1': x}
# print(example)
label_list2 = raw_datasets["train"].unique("label")
label_list2.sort()
# print(label_list2)
batch2 = tokenize_function2(example)
# print(batch2)Qh
batch2 = {k: torch.tensor(v).to(device).reshape([1, -1]) for k, v in batch2.items()}
# print(batch2['input_ids'].size())
# print(batch2['token_type_ids'].size())
# print(batch2['attention_mask'].size())
# print(batch2['labels'].size())
output2 = model2(**batch2)
model2.eval()
preds2 = []
probs2 = []
logits2 = output2.logits
prob2 = torch.softmax(logits2, dim=-1)
predictions2 = torch.argmax(logits2, dim=-1)
preds2.append(predictions2)
probs2.append(prob2)
probs2 = torch.cat(probs2, dim=0).cpu().detach().numpy()
preds2 = torch.cat(preds2, dim=-1).cpu().detach().numpy()

preds1 = torch.cat(preds, dim=0).cpu().numpy().flatten()

print("intent classification result:", end="")
print(preds2)

print("entity recognition result:", end="")
for p in range(len(x.replace(" ", "")) + 1):
    if p == 0:
        continue
    print(label_list1[int(preds1[p - 1])], end=" ")
