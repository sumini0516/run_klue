from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification, AutoTokenizer, logging
import torch
import warnings
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

sentence1 = ["토를 했어요"]
sentence2 = {"귀 안쪽 충혈": ["귀 안쪽 충혈",
                         "귀 안쪽이 빨개짐",
                         "귀 안쪽이 빨개요",
                         "귀 안쪽이 빨개졌어요",
                         "귀가 빨개졌어요",
                         "귀가 빨개요",
                         "귀 안쪽이 빨",
                         "귀가 빨",
                         "귀 안쪽에 피",
                         "귀 안에 충혈",
                         "귀 안 충혈",
                         "귀 안쪽 빨",
                         "귀 안에 피",
                         "귀 안 피"]}

# with open("symptom_voca.pickle", "rb") as fr:
#     sentence3 = pickle.load(fr)

sentence2_values = list(sentence2.values())
print(sentence2_values)
model = SentenceTransformer('klue/roberta-base')
sentence_embeddings1 = model.encode(sentence1)
sentence_embeddings2 = model.encode(sentence2_values[0])
sentence_embeddings2 = sentence_embeddings2.mean(axis=0).reshape(1, -1)
print(sentence_embeddings1.shape)
print(sentence_embeddings2.shape)
print(cosine_similarity([sentence_embeddings1[0]], sentence_embeddings2[0]))
# print(cosine_similarity([sentence_embeddings1[0]], sentence_embeddings2[0:]).argmax())
# print(sentence2[cosine_similarity([sentence_embeddings1[0]], sentence_embeddings2[0:]).argmax()])
# idxes = cosine_similarity([sentence_embeddings1[0]], sentence_embeddings2[0:]).argsort()[0][::-1][:5]
# print(idxes)
# print([sentence2[idx] for idx in idxes])
