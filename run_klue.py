from data_loader import data_loaders
from eval_metric import metrics
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer
from tokenization_kobert import KoBertTokenizer
import torch
import logging
import os
from tqdm.auto import tqdm
from transformers import get_scheduler
from transformers import AdamW

import sys

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logger = logging.getLogger(__name__)

task = sys.argv[1] if len(sys.argv) > 1 else 'myner'

lr = float(sys.argv[2]) if len(sys.argv) > 2 else 3e-5
checkpoint = sys.argv[3] if len(sys.argv) > 3 else "klue/roberta-base"  # klue/roberta-base, klue/roberta-large, monologg/kobert, gogamza/kobart-base-v2 klue/roberta-large
batch_size = int(sys.argv[4]) if len(sys.argv) > 4 else 8

print(sys.argv[0])
print(task)
print(lr)
print(checkpoint)
print(batch_size)

if task not in ['ynat', 'nli', 'sts', 're', 'ner', 'myynat', 'myner']:
    exit()

# train_dataloader, eval_dataloader = data_loaders[task]()

model_classes = {
    'ynat': AutoModelForSequenceClassification,
    'nli': AutoModelForSequenceClassification,
    'sts': AutoModelForSequenceClassification,
    're': AutoModelForSequenceClassification,
    'ner': AutoModelForTokenClassification,
    'myynat': AutoModelForSequenceClassification,
    'myner': AutoModelForTokenClassification
}

num_classes = {
    'ynat': 7,
    'nli': 3,
    'sts': 2,
    're': 30,
    'ner': 13,
    'myynat': 2,
    'myner': 17  # 411,7
}

num_epoch_task = {
    'ynat': 3,
    'nli': 1,
    'sts': 3,
    're': 3,
    'ner': 3,
    'myynat': 1,
    'myner': 10
}


# def freeze(model):
#     for name, param in model.named_parameters():
#         print("layer name:\t", name)
#         if name.find("classifier") == -1:
#             param.requires_grad = False
#             print("frozen layer:\t", name)


def train(model, optimizer, lr_scheduler, train_dataloader, eval_dataloader, num_epochs, num_training_steps, metric,
          device):
    progress_bar = tqdm(range(num_training_steps))

    model.train()
    max = 0.0
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            #print(batch)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            if step % 100 == 0 and step != 0:
                model.eval()
                performance = eval(model, eval_dataloader, metric, device)
                logger.info(f"accuracy: {performance}")

                if performance > max:
                    max = performance
                    best_ckpt = './run/model-{}.ckpt'.format(step)
                    torch.save(model, best_ckpt)
        save_path = 'D:/save_model/ner/roberta_large/'
        torch.save(model.state_dict(), save_path + "model1.pth")

    return best_ckpt


def eval(model, eval_dataloader, metric, device):
    model.eval()
    preds = []
    targets = []
    probs = []
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        prob = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)
        preds.append(predictions)
        targets.append(batch["labels"])
        probs.append(prob)

    if task in ['ner']:
        preds = torch.cat(preds, dim=0).cpu().numpy().flatten()
        targets = torch.cat(targets, dim=0).cpu().numpy().flatten()
        preds = preds[targets != -100]
        targets = targets[targets != -100]
        # print(preds)
        # print(targets)
        # for k, v in metric(preds, targets).items():
        #     print(k, v)
    elif task in ['myner']:
        preds = torch.cat(preds, dim=0).cpu().numpy().flatten()
        targets = torch.cat(targets, dim=0).cpu().numpy().flatten()
        preds = preds[targets != -100]
        targets = targets[targets != -100]
        # print(preds)
        # print(targets)
        for k, v in metric(preds, targets).items():
            print(k, v)
    else:
        probs = torch.cat(probs, dim=0).cpu().numpy()
        preds = torch.cat(preds, dim=-1).cpu().numpy()
        targets = torch.cat(targets, dim=-1).cpu().numpy()
        print(preds)
        print(targets)
        if task in ['re']:
            for k, v in metric(probs, preds, targets).items():
                print(k, v)
        else:
            for k, v in metric(preds, targets).items():
                print(k, v)

    return v


def main():
    if checkpoint in ['kobert', "monologg/kobert", 'distilkobert', 'monologg/distilkobert', "monologg/kobert-lm"]:
        tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert")
    else:
        tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')

    if task in ['myynat', 'ner', 'myner']:
        train_dataloader, eval_dataloader, test_dataloader = data_loaders[task](tokenizer, batch_size)
    else:
        train_dataloader, eval_dataloader = data_loaders[task](tokenizer, batch_size)
    model = model_classes[task].from_pretrained(checkpoint, num_labels=num_classes[task])

    metric = metrics[task]

    if task in ['re']:
        model.resize_token_embeddings(len(tokenizer))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # freeze(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    num_epochs = num_epoch_task[task]
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    best_ckpt = train(model, optimizer, lr_scheduler, train_dataloader, eval_dataloader, num_epochs, num_training_steps,
                      metric, device)
    print()

    model = torch.load(best_ckpt)

    final_performance = eval(model, test_dataloader, metric, device)
    print(final_performance)


if __name__ == '__main__':
    main()
