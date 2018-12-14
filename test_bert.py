import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm as tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

from sklearn.model_selection import train_test_split

import re
import os
import argparse

from nltk.corpus import stopwords
import nltk
#nltk.download('stopwords')
import copy
# from model import BERT_Classifier
from dataset import *
from preprocess import english_clean_series, chinese_clean_series
from model import BERT_Classifier

from collections import defaultdict
from sklearn.model_selection import KFold
import pickle
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.optimization import BertAdam



# from train_bert import BERT_Classifier, chinese_clean_series, english_clean_series
test_df = pd.read_csv("data/test.csv")

with open('save/fixed_dic.pickle', mode='rb') as f:
    fixed_dic = pickle.load(f)
print("load fixed dic.", len(fixed_dic))
#
# test_tid1 = list(test_df["tid1"])
# test_tid2 = list(test_df["tid2"])
#
# id_ = test_df["id"]
#
#
# preded_id_label = []
#
# given, not_given = 0, 0
# # 予測できるラベルもあるよね.
# #
# # fixed_dicの枝張りを収束まで行う.
#
# agree_dic = defaultdict(list)
# disagree_dic = defaultdict(list)
#
# for id1, id_label_list in fixed_dic.items():
#     if len(id_label_list) == 0:
#         continue
#     id_list = np.array(id_label_list)[:,0]
#     label_list = np.array(id_label_list)[:,1]
#     for id2, label in zip(id_list, label_list):
#         if label == 1:
#             agree_dic[id1].append(id2)
#         elif label == 2:
#             disagree_dic[id1].append(id2)
#
# print(len(agree_dic), len(disagree_dic))
#
# #番兵
# #agreeのagreeはagree. agreeのdisagreeはdisagree.
# print("二部グラフの作成...")
# change=0
# while True:
#     for tid1, agree_id_list in agree_dic.items():
#         for tid2 in agree_id_list:
#             disagree_to_tid2 = disagree_dic[tid2]
#             for dis in disagree_to_tid2:
#                 if not dis in disagree_dic[tid1]:
#                     disagree_dic[tid1].append(dis)
#                     change+=1
#
#                 if not tid1 in disagree_dic[dis]:
#                     disagree_dic[dis].append(tid1)
#                     change+=1
#
#             agree_to_tid2 = agree_dic[tid2]
#             for dis in agree_to_tid2:
#                 if not dis in agree_dic[tid1]:
#                     agree_dic[tid1].append(dis)
#                     change+=1
#
#                 if not tid1 in agree_dic[dis]:
#                     agree_dic[dis].append(tid1)
#                     change+=1
#     for tid1, disagree_id_list in disagree_dic.items():
#         for tid2 in disagree_id_list:
#
#             agree_to_tid2 = agree_dic[tid2]
#             for dis in agree_to_tid2:
#                 if not dis in disagree_dic[tid1]:
#                     disagree_dic[tid1].append(dis)
#                     change+=1
#
#                 if not tid1 in disagree_dic[dis]:
#                     disagree_dic[dis].append(tid1)
#                     change+=1
#
#     print("change number: ", change)
#     if change == 0:
#         break
#     else:
#         break
#         change = 0
#
# mujun = 0
#
# for id1, id2, each_id in zip(test_tid1, test_tid2, id_):
#     if id1 == id2:
#         preded_id_label.append((each_id, 0))
#         continue
#     if id2 in disagree_dic[id1]:
#         #check
#         if id1 in disagree_dic[id2]:
#             preded_id_label.append((each_id, 2))
#             continue
#         else:
#             mujun+=1
#
#     elif id2 in agree_dic[id1]:
#         #check
#         if id1 in agree_dic[id2]:
#             preded_id_label.append((each_id, 1))
#         else:
#             mujun+=1
#
# del agree_dic, disagree_dic
# # preded_id_label = []
# with open('save/preded_id_label.pickle', mode='wb') as f:
#     pickle.dump(preded_id_label, f)

with open('save/preded_id_label.pickle', mode='rb') as f:
    preded_id_label = pickle.load(f)
print("予測できたもの:{}, total:{}".format(len(preded_id_label), len(test_df)))


#推論
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
print("device:", device)

max_seq_en = 50
max_seq_zh = 50
batch = 3000

print("batch", batch)

test_df["title1_zh"] =  chinese_clean_series(test_df["title1_zh"])
test_df["title2_zh"] =  chinese_clean_series(test_df["title2_zh"])
test_df["title1_en"] = english_clean_series(test_df["title1_en"])
test_df["title2_en"] = english_clean_series(test_df["title2_en"])
id_ = test_df["id"]

test1_en, test2_en = list(test_df["title1_en"]), list(test_df["title2_en"])
test1_zh, test2_zh = list(test_df["title1_zh"]), list(test_df["title2_zh"])

model_dir_en = "model/BERT/newdata_5fold/en/"
model_dir_zh = "model/BERT/newdata_5fold/zh/"

MAX_fold = 5
PATH_list_en = [os.path.join(model_dir_en, "{}fold_bert.model".format(fold)) for fold in range(1,MAX_fold+1,1)]
PATH_list_zh = [os.path.join(model_dir_zh, "{}fold_bert.model".format(fold)) for fold in range(1,MAX_fold+1,1)]

y_dummy = torch.empty(len(test1_en), dtype=torch.long).random_(5)

# tokenizer_en = BertTokenizer.from_pretrained('bert-large-uncased')
tokenizer_en = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer_zh = BertTokenizer.from_pretrained('bert-base-chinese')

test_dataset_en = BERTDataset(test1_en, test2_en, y_dummy, tokenizer_en, seq_length=max_seq_en)
test_dataset_zh = BERTDataset(test1_zh, test2_zh, y_dummy, tokenizer_zh, seq_length=max_seq_zh)

test_loader_en = DataLoader(test_dataset_en, batch_size=batch, shuffle=False)
test_loader_zh = DataLoader(test_dataset_zh, batch_size=batch, shuffle=False)

average_prediction = []

#inference english
bert_model = BertModel.from_pretrained('bert-base-uncased')
# bert_model = BertModel.from_pretrained('bert-large-uncased')

model = BERT_Classifier(bert_model)
print("inference english...")
for PATH in PATH_list_en:
    model.load_state_dict(torch.load(PATH))
    model = model.to(device)

    print("model loaded:{}".format(PATH))

    with torch.no_grad():
        model.eval()
        predictions = []
        for batch_idx, sample_batch in enumerate(tqdm(test_loader_en)):
            input_ids = sample_batch["input_ids"].to(device)
            input_mask = sample_batch["input_mask"].to(device)
            input_type_ids = sample_batch["input_type_ids"].to(device)

            output = F.softmax(model(input_ids, input_mask), dim=1)
            output = output.cpu().numpy()
            #print("model out:",output.shape)

            if batch_idx == 0:
                predictions = output
            else:
                predictions = np.vstack((predictions, output))

    average_prediction.append(predictions)



# inference chinese
print("inference chinese...")
bert_model = BertModel.from_pretrained('bert-base-chinese')
model = BERT_Classifier(bert_model, embedding_dim=768)
for PATH in PATH_list_zh:
    print(PATH)
    model.load_state_dict(torch.load(PATH))
    model.to(device)

    print("model loaded:{}".format(PATH))

    with torch.no_grad():
        model.eval()
        predictions = []
        for batch_idx, sample_batch in enumerate(tqdm(test_loader_zh)):
            input_ids = sample_batch["input_ids"].to(device)
            input_mask = sample_batch["input_mask"].to(device)
            input_type_ids = sample_batch["input_type_ids"].to(device)

            output = F.softmax(model(input_ids, input_mask), dim=1)
            output = output.cpu().numpy()

            if batch_idx == 0:
                predictions = output
            else:
                predictions = np.vstack((predictions, output))

    average_prediction.append(predictions)

average_prediction = np.array(average_prediction)
print("total prediction:", average_prediction.shape)
average_prediction = np.mean(average_prediction, axis=0)
# print("total pred:", average_prediction.shape)
print("prediction:",average_prediction)

with open('save/average_prediction.pickle', mode='wb') as f:
    pickle.dump(predictions, f)

predictions = np.argmax(average_prediction, axis=1)
print("predictions:", predictions.shape)


if len(preded_id_label) == 0:
    preded_labels = []
    preded_id = []
else:
    preded_id = np.array(preded_id_label)[:, 0]
    preded_labels = np.array(preded_id_label)[:, 1]
print("directly preded label:", len(preded_id))


fixed_predictions = []
for each_id, p in zip(id_, predictions):
    if each_id in preded_id:
        #trainの中に現れてたやつ
        idx = list(preded_id).index(each_id)
        fixed_predictions.append(preded_labels[idx])
    else:
        fixed_predictions.append(p)


#'unrelated', 0
#'agreed', 1
#'disagreed', 2

new_predictions = []
for p in fixed_predictions:
    if p == 0:
        new_predictions.append("unrelated")
    elif p==1:
        new_predictions.append("agreed")
    elif p==2:
        new_predictions.append("disagreed")


submit_csv = pd.concat([id_, pd.Series(new_predictions)], axis=1)
#display(submit_csv)

submit_csv.columns = ["Id", "Category"]
submit_csv.to_csv("result/submit.csv", header=True, index=False)
