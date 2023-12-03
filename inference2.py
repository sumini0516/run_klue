from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification, AutoTokenizer, logging
import torch
import warnings
import sys

warnings.filterwarnings('ignore')
logging.set_verbosity_error()


def program():
    print('=======================================================')
    print('*** 모델을 불러오고 있습니다... 잠시만 기다려주세요! ***')
    print('=======================================================')
    PATH1 = "./model_save/model.pth"
    PATH2 = "./model_save/model2.pth"
    pretrained_model = "klue/roberta-large"

    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    model1 = AutoModelForTokenClassification.from_pretrained("klue/roberta-large", num_labels=15).to(device)
    model2 = AutoModelForSequenceClassification.from_pretrained("klue/roberta-large", num_labels=7).to(device)
    model1.load_state_dict(torch.load(PATH1))
    model2.load_state_dict(torch.load(PATH2))

    label_list2 = ['disease_basic', 'disease_dict_clinic', 'disease_dict_expense', 'fallback', 'hospital_book',
                   'hospital_recommend', 'search_disease']

    def make_labels(example):
        strip_char = "##"
        original_clean_tokens = []
        original_clean_labels = []
        return_tokenized_word = []
        sentence = ""
        for token, tag in zip(example['tokens'], example['ner_tags']):
            sentence += token
            if token == " ":
                continue
            original_clean_tokens.append(token)
            original_clean_labels.append(tag)
        sent_words = sentence.split(" ")
        modi_labels = []
        char_idx = 0
        for word in sent_words:
            correct_syllable_num = len(word)
            tokenized_word = tokenizer.tokenize(word)
            return_tokenized_word.append(tokenized_word)
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
        example['sentence'] = sentence
        example['labels'] = modi_labels
        return example, return_tokenized_word

    def make_padding_label(example):
        l = len(example["input_ids"])
        labels = example['labels']
        labels = [-100] + labels + ([-100] * (l - len(labels) - 1))
        example['labels'] = labels
        return example

    def tokenize_function1(example):
        return tokenizer(example["sentence"], truncation=True, padding='max_length', max_length=64)

    def tokenize_function2(example):
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
        list_token = []
        list_ner_tags = []
        for token in input:
            list_token.append(token)
        for token in list_token:
            list_ner_tags.append(int(14))
        raw_datasets1["sentence"] = input
        raw_datasets1["tokens"] = list_token
        raw_datasets1["ner_tags"] = list_ner_tags
        return raw_datasets1

    def entity_prediction(pred_idx, t, word, predict):
        if pred_idx % 2:
            if word[t][:2] == '##':
                predict += word[t][2:]
            else:
                predict += ' ' + word[t]

            if predict == '':
                predict += word[t]
        else:
            if predict != '':
                predict += ', ' + word[t]
            else:
                predict += word[t]

        return predict

    while True:
        preds_symptom, preds_animal, preds_name, preds_date, preds_disease, preds_location, preds_time, intent = '', '', '', '', '', '', '', ''
        raw_datasets1 = {}
        preds2, probs2, idx = [], [], []
        try:
            print('=====================================')
            print('*** 종료를 원하실 때는 exit을 입력해주세요 ***\n')
            input_text = input("증상을 입력하세요 : ")
            if input_text.lower() == 'exit':
                break
            raw_datasets1 = preprocess1(input_text)
            label, tokenized_words = make_labels(raw_datasets1)
            print("tokenized_words:",tokenized_words)
            batch = tokenize_function1(raw_datasets1)
            batch.update(label)
            batch = make_padding_label(batch)
            batch.pop('ner_tags')
            batch.pop('sentence')
            batch.pop('tokens')
            batch2 = {k: torch.tensor(v).to(device).reshape([1, -1]) for k, v in batch.items()}
            preds1 = []
            probs1 = []
            output = model1(**batch2)
            logits = output.logits
            print("logits:",logits)
            prob = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            preds1.append(predictions)
            print("predictions:", predictions)
            probs1.append(prob)
            print("prob:", prob)
            preds1 = torch.cat(preds1, dim=0).cpu().numpy().flatten()
            for j, k in enumerate(input_text):
                if k != ' ':
                    idx.append(j)
            cnt = 0
            for idx, word in enumerate(tokenized_words):
                len_word = len(word)
                for t in range(len_word):
                    cnt += 1
                    pred_idx = preds1[cnt]
                    if pred_idx in [0, 1]:
                        preds_symptom = entity_prediction(pred_idx, t, word, preds_symptom)
                    elif pred_idx in [2, 3]:
                        preds_animal = entity_prediction(pred_idx, t, word, preds_animal)
                    elif pred_idx in [4, 5]:
                        preds_name = entity_prediction(pred_idx, t, word, preds_name)
                    elif pred_idx in [6, 7]:
                        preds_date = entity_prediction(pred_idx, t, word, preds_date)
                    elif pred_idx in [8, 9]:
                        preds_time = entity_prediction(pred_idx, t, word, preds_time)
                    elif pred_idx in [10, 11]:
                        preds_disease = entity_prediction(pred_idx, t, word, preds_disease)
                    elif pred_idx in [12, 13]:
                        preds_location = entity_prediction(pred_idx, t, word, preds_location)
            example = {'label': 6, 'sentence1': input_text}
            batch2 = tokenize_function2(example)
            batch2 = {k: torch.tensor(v).to(device).reshape([1, -1]) for k, v in batch2.items()}
            output2 = model2(**batch2)
            model2.eval()
            logits2 = output2.logits
            prob2 = torch.softmax(logits2, dim=-1)
            predictions2 = torch.argmax(logits2, dim=-1)
            preds2.append(predictions2)
            probs2.append(prob2)
            probs2 = torch.cat(probs2, dim=0).cpu().detach().numpy()
            preds2 = torch.cat(preds2, dim=-1).cpu().detach().numpy()
            intent = label_list2[preds2[0]]
            with open('./IntoCNS_database.csv', 'a', encoding='utf-8-sig') as db:
                db.write(intent)
                db.write(',')
                db.write(input_text)
                db.write(',')
                db.write(preds_name)
                db.write(',')
                db.write(preds_animal)
                db.write(',')
                db.write(preds_symptom)
                db.write(',')
                db.write(preds_date)
                db.write(',')
                db.write(preds_disease)
                db.write(',')
                db.write(preds_location)
                db.write(',')
                db.write(preds_time)
                db.write('\n')
            print('=====================================')
            print('[ Intent ] : ', intent)
            if preds_animal != '':
                print('[ Species ] : ', preds_animal)
            if preds_name != '':
                print('[ Name ] : ', preds_name)
            if preds_symptom != '':
                print('[ Symptom ] : ', preds_symptom)
            if preds_disease != '':
                print('[ Disease ] : ', preds_disease)
            if preds_date != '':
                print('[ Date ] : ', preds_date)
            if preds_location != '':
                print('[ Location ] : ', preds_location)
            if preds_time != '':
                print('[ Time ] : ', preds_time)
        except EOFError:
            break
    print('\n\n감사합니다. 재시작을 원하실 경우 프로그램을 다시 실행시켜주세요.\n')
    return


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    program()
    warnings.filterwarnings('ignore')
    sys.exit()
