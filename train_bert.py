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
from collections import defaultdict
import copy

from dataset import *
from preprocess import english_clean_series, chinese_clean_series
from pseudo_labeling import pseudo_label_test
from model import BERT_Classifier

from sklearn.model_selection import KFold, StratifiedKFold

from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.optimization import BertAdam

import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
## Required parameters
parser.add_argument("-f", "--fold",
                    default=-1,
                    type=int,
                    required=False,
                    help="the number of k-fold.")
parser.add_argument("-g", "--gpu_id",
                    default=-1,
                    type=int,
                    required=False,
                    help="The ID of using GPU.")
parser.add_argument("-b", "--batch",
                    default=8,
                    type=int,
                    required=False,
                    help="The sample size of mini-batch.")
parser.add_argument("-l", "--language",
                    default="en",
                    type=str,
                    required=True,
                    help="language to train BERT.")


args = parser.parse_args()

if args.gpu_id < 0 and not torch.cuda.is_available():
    device = "cpu"
elif  args.gpu_id >= 0 and torch.cuda.is_available():
    device = "cuda:{}".format(args.gpu_id)
else:
    device = "cuda:0"


print("----------train_bert_zh.py starting--------------")
print("language:", args.language)
print("device:", device)
print("fold:", args.fold)




EMBEDDING_DIM = 512
HIDDEN_DIM = 256
max_seq_en = 50
max_seq_zh = 50
EPOCH=100
gradient_accumulation_steps=1
batch=args.batch
local_rank=-1
learning_rate=5e-5
warmup_proportion=0.1

pseudo_test = True


#device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("EPOCH", EPOCH)
print("batch", batch)

train_df = pd.read_csv("data/train.csv")
train_df.replace('unrelated', 0, inplace=True)
train_df.replace('agreed', 1, inplace=True)
train_df.replace('disagreed', 2, inplace=True)


train_df["title1_zh"] =  chinese_clean_series(train_df["title1_zh"])
train_df["title2_zh"] =  chinese_clean_series(train_df["title2_zh"])
train_df["title1_en"] = english_clean_series(train_df["title1_en"])
train_df["title2_en"] = english_clean_series(train_df["title2_en"])

#Shuffle dataframe.
train_df = train_df.sample(frac=1, random_state=0).reset_index(drop=True)#.iloc[:300, :]


# K-Fold Cross validation
fold_num = 5
kf = StratifiedKFold(n_splits=fold_num, random_state=42)
kf.get_n_splits(train_df)


# Pseudo labeling to Test dataset.
if pseudo_test:
    test_pseudo = pseudo_label_test(threshold=0.99)

train_data_list = []
val_data_list = []
for train_index, val_index in kf.split(train_df, train_df["label"]):
    training_df = train_df.iloc[train_index]
    val_df = train_df.iloc[val_index]

    train1_en, train2_en = list(training_df["title1_en"]), list(training_df["title2_en"])
    train1_zh, train2_zh = list(training_df["title1_zh"]), list(training_df["title2_zh"])

    y_train = list(training_df["label"])

    val1_en, val2_en = list(val_df["title1_en"]), list(val_df["title2_en"])
    val1_zh, val2_zh = list(val_df["title1_zh"]), list(val_df["title2_zh"])
    y_val = list(val_df["label"])

    for data in test_pseudo:
        (en1, en2, zh1, zh2, max_label) = data
        train1_en.append(en1)
        train2_en.append(en2)
        train1_zh.append(zh1)
        train2_zh.append(zh2)
        y_train.append(max_label)

    train_data_list.append((train1_en,train2_en,train1_zh,train2_zh, y_train))
    val_data_list.append((val1_en, val2_en,val1_zh, val2_zh, y_val))


# K-foldを分割して行う場合
if args.fold < 0 :
    split_kfold = False
    fold = 1
else:
    split_kfold = True
    fold = args.fold

fold_count = 1
reach_target_fold = False

for train_fold, val_fold in zip(train_data_list,val_data_list):

    if split_kfold:
        if not fold_count == fold:
            fold_count += 1
            continue
        else:
            reach_target_fold = True

    print("{}/{} fold :".format(fold, fold_num))
    print("train length:{}, val length:{}".format(len(train_fold[0]), len(val_fold[0])))

    (train1_en,train2_en,train1_zh,train2_zh, y_train) = train_fold
    (val1_en, val2_en,val1_zh, val2_zh, y_val) = val_fold

    c = Counter(y_train)
    class_weight = []
    for label, num in sorted(c.items()):
        print(label, num)
        class_weight.append(len(y_train)/(3*num))
    class_weight = torch.FloatTensor(class_weight).to(device)

    num_train_steps = int(len(train_fold[0]) / batch / gradient_accumulation_steps * EPOCH)

    if args.language=="zh":
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        bert_model = BertModel.from_pretrained('bert-base-chinese').to(device)
        train_dataset = BERTDataset(train1_zh, train2_zh, y_train, tokenizer, seq_length=max_seq_zh)
        val_dataset = BERTDataset(val1_zh, val2_zh, y_val, tokenizer, seq_length=max_seq_zh)
    elif args.language=="en":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
        train_dataset = BERTDataset(train1_en, train2_en, y_train, tokenizer, seq_length=max_seq_en)
        val_dataset = BERTDataset(val1_en, val2_en, y_val, tokenizer, seq_length=max_seq_en)
    else:
        print("choose either en or zh as language.")
    model = BERT_Classifier(bert_model).to(device)

    # print("model parameters", [j, i.size() for j, i in model.named_parameters()])
    # for n, p in model.named_parameters():
    #     print(n)
    #     # print(p)
    #     print("--------")

    loss_function = nn.CrossEntropyLoss()
    weighted_loss_function = nn.CrossEntropyLoss(weight=class_weight)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
    t_total = num_train_steps
    if local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=learning_rate,
                         warmup=warmup_proportion,
                         t_total=t_total)

    #ミニバッチ内のクラス比を揃える.
    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=False, sampler=sampler)#, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)

    def train(epoch, global_step):
        model.train()
        tr_loss = 0
        for batch_idx, sample_batch in enumerate(tqdm(train_loader)):
            input_ids = sample_batch["input_ids"].to(device)
            input_mask = sample_batch["input_mask"].to(device)
            input_type_ids = sample_batch["input_type_ids"].to(device)
            y = sample_batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, input_mask)

            loss = loss_function(outputs, y)
            tr_loss += loss.item()
            loss.backward()
            optimizer.step()

            global_step+=1

            if batch_idx%100==0:
                print("==========>train_loss:{}".format(loss))


        print("epoch:{},train_loss:{:.4f}".format(epoch+1 ,loss))

        tr_loss /= (batch_idx+1)
        return model, tr_loss, global_step

    def test(tr_loss):
        with torch.no_grad():
            model.eval()
            test_loss = 0
            correct = 0

            for batch_idx, sample_batch in enumerate(val_loader):
                input_ids = sample_batch["input_ids"].to(device)
                input_mask = sample_batch["input_mask"].to(device)
                input_type_ids = sample_batch["input_type_ids"].to(device)
                y = sample_batch["label"].to(device)

                output = model(input_ids, input_mask)
                # sum up batch loss
                test_loss += weighted_loss_function(output, y).item()

                # get the index of the max log-probability
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(y.view_as(pred)).sum().item()

            test_loss /= batch_idx+1

            accuracy = weighted_accuracy(pred, y)

            print('Validation set: Weighted loss: {:.4f}, Weighted Accuracy: {}/{} ({:.2f}%)'
                  .format(test_loss, correct, len(val_loader.dataset),
                          accuracy))

            result = {'val_loss': test_loss,
                      'eval_accuracy': accuracy,
                      'global_step': global_step,
                      'train_loss': tr_loss}

            output_eval_file = os.path.join("result/test/{}".format(args.language), "{}_fold_eval_results.txt".format(fold))
            with open(output_eval_file, "a") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

            return test_loss, accuracy


    def weighted_accuracy(pred, true):
        true = true.cpu().numpy()
        pred = pred.cpu().numpy()
        class_weight = [1/16, 1/15, 1/5]
        score = 0
        perfect_score = 0

        for p, t in zip(true, pred):
            if p == t:
                if t == 0:
                    score += 1/16
                    perfect_score += 1/16
                elif t == 1:
                    score += 1/15
                    perfect_score += 1/15
                elif t == 2:
                    score += 1/5
                    perfect_score += 1/5
            else:
                if t == 0:
                    perfect_score += 1/16
                elif t == 1:
                    perfect_score += 1/15
                elif t == 2:
                    perfect_score += 1/5
        #print("score:{}, ideal:{}".format(score, perfect_score))
        return 100 * score/perfect_score

    def save_model(model, val_accuracy, save_path="model/BERT/test/{}/".format(args.language)):
        name = "{}fold_bert.model".format(fold)
        PATH = os.path.join(save_path, name)
        torch.save(model.cpu().state_dict(), PATH)

    lowest_loss = 1000000000
    highest_accuracy = 0
    global_step=0
    loss_when_best = 1000000000

    for epoch in range(EPOCH):

        model, tr_loss, global_step = train(epoch, global_step)
        val_loss, accuracy = test(tr_loss)

        if accuracy >= highest_accuracy:
            if accuracy == highest_accuracy:
                if loss_when_best > val_loss:
                    save_model(model, highest_accuracy)
                    loss_when_best = val_loss
            else:
                highest_accuracy = accuracy
                save_model(model, highest_accuracy)
                loss_when_best = val_loss
        print("highest_accuracy:{:.2f}% \n".format(highest_accuracy))
        output_eval_file = os.path.join("result/test/{}".format(args.language), "{}_fold_eval_results.txt".format(fold))
        with open(output_eval_file, "a") as writer:
            writer.write("highest_accuracy:{}\n".format(highest_accuracy))
            writer.write("\n")

    if split_kfold and reach_target_fold:
        break

    fold+=1
