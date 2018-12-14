import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm as tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

import re

from nltk.corpus import stopwords
import pickle
from collections import defaultdict
import copy


def english_clean_series(series):
    # 大文字--->小文字
    series = series.str.lower()

    def clean_seq(seq):
        ori = copy.copy(seq)

        seq = seq.replace("it's", "it is")
        seq = seq.replace("he's", "he is")
        seq = seq.replace("she's", "she is")
        seq = seq.replace("you're", "you are")
        seq = seq.replace("we're", "we are")
        seq = seq.replace("they're", "they are")
        seq = seq.replace("i'm", "i am")
        seq = seq.replace("don't", "do not")
        seq = seq.replace("does't", "does not")
        seq = seq.replace("didn't", "did not")
        seq = seq.replace("aren't", "are not")
        seq = seq.replace("weren't", "were not")
        seq = seq.replace("isn't", "is not")
        seq = seq.replace("wasn't", "was not")
        seq = seq.replace("haven't", "have not")
        seq = seq.replace("hasn't", "has not")
        seq = seq.replace("can't", "can not")
        seq = seq.replace("cannot", "can not")

        seq = seq.replace("shouldn't", "should not")
        seq = seq.replace("wouldn't", "would not")
        seq = seq.replace("couldn't", "could not")
        seq = seq.replace("mightn't", "might not")
        seq = seq.replace("mustn't", "must not")
        seq = seq.replace("needn't", "need not")
        seq = seq.replace("won't", "will not")

        seq = seq.replace("\n", "")

        seq = seq.replace("< i >", "")
        seq = seq.replace("< / i >", "")

        seq = re.sub(r'#[,."''“”。、#()→⇒←↓↑:;㊙️【《》=|/+<>]+', '', seq)
        seq = re.sub(r'[!?]+', ' ', seq)
        # seq = re.sub(r'[$]+', '$ ', seq)
        # seq = re.sub(r'[0-9]+', '<NUM>', seq)

        if len(seq)==0:
            print("0 lengrh assert!!,",ori, seq)
        return seq

    series = series.apply(clean_seq)
    return series

def chinese_clean_series(series):
    def clean_seq(seq):
        seq = str(seq)
        ori = copy.copy(seq)

        seq = seq.replace("< i >", "")
        seq = seq.replace("< / i >", "")
        seq = seq.replace("\n", "")
        seq = re.sub(r'#[,."''“”。、#()→⇒←↓↑:;_㊙️【《》=|/<>]+', '', seq)
        seq = re.sub(r'[!！？?-]+', ' ', seq)
        seq = re.sub(r'[$]+', '$ ', seq)
        # seq = re.sub(r'[0-9]+', '<NUM>', seq)

        if len(seq)==0:
            print("0 lengrh assert!!,",ori, seq)

        return seq

    series = series.apply(clean_seq)
    return series




def estimate_relationship(df):
    labels = list(df["label"])
    id1_train = list(df["tid1"])
    id2_train = list(df["tid2"])

    given_dic = defaultdict(list)
    fixed_dic = defaultdict(list)

    # agree, disagree辞書の作成
    # fixed_dicではA,B,labelがきた時、B,A,labelも登録する.
    # ラベルの修正もする. agree とunrelatedの衝突ではagreeの勝ち,agreeとdisagreeではdisagreeの勝ち.
    for idx, id1 in enumerate(id1_train):
        label = labels[idx]
        id2 = id2_train[idx]

        if not len(fixed_dic[id1]) == 0:
            already_given_id = np.array(fixed_dic[id1])[:,0]
            already_given_label = np.array(fixed_dic[id1])[:,1]
            if not id2 in already_given_id:
                #まだ登録されていないとき
                fixed_dic[id1].append([id2, label])
            else:
                #すでに登録ずみのとき
                id2_idx = list(already_given_id).index(id2)
                already_given = already_given_label[id2_idx]
                if not label == already_given:
                    # tid1,A, tid2,B : label 1 だが tid1,B, tid2,A : label 0というケースあり.  trainで88件.
                    # その場合は修正.
                    #print(id1, id2, already_given, label)
                    if label == 0:
                        pass
                    elif label == 1 and already_given == 0:
                        true_label = 1
                        #ラベルの修正.
                        fixed_dic[id1][id2_idx][1] = true_label
                    elif label == 2 or already_given == 2:
                        true_label = 2
                        fixed_dic[id1][id2_idx][1] = true_label

                    #print(id1, given_dic[id1][id2_idx])

                else:
                    pass

        else:
            #最初に登録するとき
            fixed_dic[id1].append([id2, label])

        if not len(fixed_dic[id2]) == 0:
            already_given_id = np.array(fixed_dic[id2])[:,0]
            already_given_label = np.array(fixed_dic[id2])[:,1]
            if not id1 in already_given_id:
                fixed_dic[id2].append([id1, label])
            else:
                id1_idx = list(already_given_id).index(id1)
                already_given = already_given_label[id1_idx]
                if not label == already_given:
                    #print(id1, id2, label, already_given_label [id1_idx])
                    # tid1,A, tid2,B : label 1 だが tid1,B, tid2,A : label 0というケースあり.  trainで88件.
                    if label == 0:
                        pass
                    elif label == 1 and already_given == 0:
                        true_label = 1
                        fixed_dic[id2][id1_idx][1] = true_label
                    elif label == 2 or already_given == 2:
                        true_label = 2
                        fixed_dic[id2][id1_idx][1] = true_label


        else:
            #最初に登録するとき
            fixed_dic[id2].append([id1, label])



    return 0

def make_new_data(df):

    title1_en = list(df["title1_en"])
    title2_en = list(df["title2_en"])
    title1_zh = list(df["title1_zh"])
    title2_zh = list(df["title2_zh"])
    labels = list(df["label"])
    id1_train = list(df["tid1"])
    id2_train = list(df["tid2"])


    # id-text dictionary
    id_to_text_en = defaultdict(list)
    id_to_text_zh = defaultdict(list)
    for idx, id1 in enumerate(id1_train):
        #if not id1 in id_to_text_en.keys():
        id_to_text_en[id1] = title1_en[idx]
        id_to_text_zh[id1] = title1_zh[idx]

    for idx, id2 in enumerate(id2_train):
        #if not id2 in id_to_text_en.keys():
        id_to_text_en[id2] = title2_en[idx]
        id_to_text_zh[id2] = title2_zh[idx]


    # key : id,
    # value : id of agreed text or diagreed text.
    agree_dic = defaultdict(list)
    disagree_dic = defaultdict(list)
    #すでにラベルが与えられているid-idを保持する.
    given_dic = defaultdict(list)
    bidirection_dic = defaultdict(list)

    fixed_dic = defaultdict(list)


    # given_dicは単純に出てきた関係を保持
    for idx, id1 in enumerate(id1_train):
        label = labels[idx]
        id2 = id2_train[idx]
        given_dic[id1].append((id2, label))

    # for idx, id1 in enumerate(id1_train):
    #     label = labels[idx]
    #     id2 = id2_train[idx]
    #     given_dic[id1].append((id2, label))


    # agree, disagree辞書の作成
    # fixed_dicではA,B,labelがきた時、B,A,labelも登録する.
    # ラベルの修正もする. agree とunrelatedの衝突ではagreeの勝ち,agreeとdisagreeではdisagreeの勝ち.
    for idx, id1 in enumerate(id1_train):
        label = labels[idx]
        id2 = id2_train[idx]

        if not len(fixed_dic[id1]) == 0:
            already_given_id = np.array(fixed_dic[id1])[:,0]
            already_given_label = np.array(fixed_dic[id1])[:,1]
            if not id2 in already_given_id:
                #まだ登録されていないとき
                fixed_dic[id1].append([id2, label])
            else:
                #すでに登録ずみのとき
                id2_idx = list(already_given_id).index(id2)
                already_given = already_given_label[id2_idx]
                if not label == already_given:
                    # tid1,A, tid2,B : label 1 だが tid1,B, tid2,A : label 0というケースあり.  trainで88件.
                    # その場合は修正.
                    #print(id1, id2, already_given, label)
                    if label == 0:
                        pass
                    elif label == 1 and already_given == 0:
                        true_label = 1
                        #ラベルの修正.
                        fixed_dic[id1][id2_idx][1] = true_label
                    elif label == 2 or already_given == 2:
                        true_label = 2
                        fixed_dic[id1][id2_idx][1] = true_label

                    #print(id1, given_dic[id1][id2_idx])

                else:
                    pass

        else:
            #最初に登録するとき
            fixed_dic[id1].append([id2, label])

        if not len(fixed_dic[id2]) == 0:
            already_given_id = np.array(fixed_dic[id2])[:,0]
            already_given_label = np.array(fixed_dic[id2])[:,1]
            if not id1 in already_given_id:
                fixed_dic[id2].append([id1, label])
            else:
                id1_idx = list(already_given_id).index(id1)
                already_given = already_given_label[id1_idx]
                if not label == already_given:
                    #print(id1, id2, label, already_given_label [id1_idx])
                    # tid1,A, tid2,B : label 1 だが tid1,B, tid2,A : label 0というケースあり.  trainで88件.
                    if label == 0:
                        pass
                    elif label == 1 and already_given == 0:
                        true_label = 1
                        fixed_dic[id2][id1_idx][1] = true_label
                    elif label == 2 or already_given == 2:
                        true_label = 2
                        fixed_dic[id2][id1_idx][1] = true_label


        else:
            #最初に登録するとき
            fixed_dic[id2].append([id1, label])

    # with open('save/fixed_dic.pickle', mode='wb') as f:
    #     pickle.dump(fixed_dic, f)
    # print("saved fixed_dic")


    #print("agree dic:{}, disagree dic:{}".format(len(agree_dic), len(disagree_dic)))

    # 前後入れ替え重複の除去
    fixed_dic_cleaned = copy.deepcopy(fixed_dic)
    print("重複の除去...")
    for id_, id_label_list in fixed_dic_cleaned.items():
        #print(id_label_list)
        if len(id_label_list) == 0:
            continue
        id_list = np.array(id_label_list)[:,0]
        for eachid in id_list:
            id_label_list2 = fixed_dic_cleaned[eachid]
            if len(id_label_list2) == 0:
                continue

            id_list2 = list(np.array(id_label_list2)[:,0])
            if id_ in id_list2:
                idx = list(id_list2).index(id_)
                id_label_list2.pop(idx)




    # for id1, id_label_list in fixed_dic.items():
    #     if len(id_label_list) == 0:
    #         continue
    #     id_list = np.array(id_label_list)[:,0]
    #     label_list = np.array(id_label_list)[:,1]
    #     for id2, label in zip(id_list, label_list):
    #
    #         if label == 1:
    #             agree_dic[id1].append(id2)
    #         elif label == 2:
    #             disagree_dic[id1].append(id2)


    new_data = []
    given_label_agree = []
    given_label_dis = []


    for id1, id_label_list in fixed_dic_cleaned.items():
        if len(id_label_list) == 0:
            continue
        id2_list = np.array(id_label_list)[:,0]
        label_list = np.array(id_label_list)[:,1]

        if len(given_dic[id1]) == 0:
            continue
        given_id2_list = np.array(given_dic[id1])[:, 0]
        given_label_list = np.array(given_dic[id1])[:, 1]

        for id2, label in zip(id2_list, label_list):
            if id2 in given_id2_list:
                # givenされてるもの.
                continue

            new_data.append((id_to_text_en[id1], id_to_text_en[id2], id_to_text_zh[id1], id_to_text_zh[id2], int(label)))

    #
    # print("fixed data length:{}, original:{}".format(len(new_data), len(id1_train)))


    # givenラベルから予測されるラベルをもつ.
    # A-B A-CならB-Cだろう.

    # forecast_dic = defaultdict(list)
    #
    # for id_, agree_ids in agree_dic.items():
    #     disagree_ids = disagree_dic[id_]
    #
    #     for agree_id in agree_ids:
    #         given_ids_labels= fixed_dic[agree_id]
    #         if len(given_ids_labels) == 0:
    #             continue
    #         given_ids = np.array(given_ids_labels)[:, 0]
    #         given_labels = np.array(given_ids_labels)[:, 1]
    #         assert given_ids.shape == given_labels.shape
    #
    #         # new 'disagree data'
    #         for disagree_id in disagree_ids:
    #             if disagree_id in given_ids:
    #                 #すでにラベルが与えられている時.
    #                 idx = list(given_ids).index(disagree_id)
    #                 label = given_labels[idx]
    #                 given_label_dis.append(label)
    #                 pass
    #             else:
    #                 #ラベルが陽に与えられていないとき.
    #                 forecast_dic[agree_id].append((disagree_id, 2))
    #                 forecast_dic[disagree_id].append((agree_id, 2))
    #                 new_data.append((id_to_text_en[agree_id], id_to_text_en[disagree_id], id_to_text_zh[agree_id], id_to_text_zh[disagree_id], 2))
    #
    #         # new 'agree data'
    #         for agree_id2 in agree_ids:
    #             if agree_id == agree_id2:
    #                 continue
    #             else:
    #                 if agree_id2 in given_ids:
    #                     #すでにラベルが与えられている時.
    #                     idx = list(given_ids).index(agree_id2)
    #                     label = given_labels[idx]
    #                     given_label_agree.append(label)
    #                     pass
    #                 else:
    #                     pass
    #                     #ラベルが陽に与えられていないとき.
    #                     forecast_dic[agree_id].append((agree_id2, 1))
    #                     forecast_dic[agree_id2].append((agree_id, 1))
    #
    #                     # new_data.append((id_to_text_en[agree_id], id_to_text_en[agree_id2], id_to_text_zh[agree_id], id_to_text_zh[agree_id2], 1))

#     c = Counter(given_label_agree)
#     print("given_label_agree", c)
#     c = Counter(given_label_dis)
#     print("given_label_disagree", c)
    print("final data length:",len(new_data))
    # with open('save/fixed_dic.pickle', mode='wb') as f:
    #     pickle.dump(fixed_dic, f)
    # with open('save/given_dic.pickle', mode='wb') as f:
    #     pickle.dump(given_dic, f)
    # with open('save/forecast_dic.pickle', mode='wb') as f:
    #     pickle.dump(forecast_dic, f)

    return new_data#, given_dic, fixed_dic, forecast_dic


def clean_df(df):


    return df



def preprocess_():

    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    #
    # def english_clean_series(series):
    #     # 大文字--->小文字
    #     series = series.str.lower()
    #
    #     def clean_seq(seq):
    #         ori = copy.copy(seq)
    #
    #         seq = seq.replace("it's", "it is")
    #         seq = seq.replace("he's", "he is")
    #         seq = seq.replace("she's", "she is")
    #         seq = seq.replace("you're", "you are")
    #         seq = seq.replace("we're", "we are")
    #         seq = seq.replace("they're", "they are")
    #         seq = seq.replace("i'm", "i am")
    #         seq = seq.replace("don't", "do not")
    #         seq = seq.replace("does't", "does not")
    #         seq = seq.replace("didn't", "did not")
    #         seq = seq.replace("aren't", "are not")
    #         seq = seq.replace("weren't", "were not")
    #         seq = seq.replace("isn't", "is not")
    #         seq = seq.replace("wasn't", "was not")
    #         seq = seq.replace("haven't", "have not")
    #         seq = seq.replace("hasn't", "has not")
    #         seq = seq.replace("can't", "can not")
    #         seq = seq.replace("cannot", "can not")
    #
    #         seq = seq.replace("shouldn't", "should not")
    #         seq = seq.replace("wouldn't", "would not")
    #         seq = seq.replace("couldn't", "could not")
    #         seq = seq.replace("mightn't", "might not")
    #         seq = seq.replace("mustn't", "must not")
    #         seq = seq.replace("needn't", "need not")
    #         seq = seq.replace("won't", "will not")
    #
    #         seq = seq.replace("\n", "")
    #
    #         seq = seq.replace("< i >", "")
    #         seq = seq.replace("< / i >", "")
    #
    #         seq = re.sub(r'[,."''“”。、#()→⇒←↓↑:;㊙️【《》=|/+<>]+', '', seq)
    #         seq = re.sub(r'[!?]+', ' ', seq)
    #         # seq = re.sub(r'[$]+', '$ ', seq)
    #         # seq = re.sub(r'[0-9]+', '<NUM>', seq)
    #
    #         if len(seq)==0:
    #             print("0 lengrh assert!!,",ori, seq)
    #         return seq
    #
    #     series = series.apply(clean_seq)
    #     return series
    #
    # def chinese_clean_series(series):
    #     def clean_seq(seq):
    #         seq = str(seq)
    #         ori = copy.copy(seq)
    #
    #         seq = seq.replace("< i >", "")
    #         seq = seq.replace("< / i >", "")
    #         seq = seq.replace("\n", "")
    #         seq = re.sub(r'[,."''“”。、#()→⇒←↓↑:;_㊙️【《》=|/<>]+', '', seq)
    #         seq = re.sub(r'[!！？?-]+', ' ', seq)
    #         seq = re.sub(r'[$]+', '$ ', seq)
    #         # seq = re.sub(r'[0-9]+', '<NUM>', seq)
    #
    #         if len(seq)==0:
    #             print("0 lengrh assert!!,",ori, seq)
    #
    #         return seq
    #
    #     series = series.apply(clean_seq)
    #     return series

    train_df["title1_en"] = english_clean_series(train_df["title1_en"])
    train_df["title2_en"] = english_clean_series(train_df["title2_en"])
    train_df["title1_zh"] =  chinese_clean_series(train_df["title1_zh"])
    train_df["title2_zh"] =  chinese_clean_series(train_df["title2_zh"])

    test_df["title1_en"] = english_clean_series(test_df["title1_en"])
    test_df["title2_en"] = english_clean_series(test_df["title2_en"])
    test_df["title1_zh"] =  chinese_clean_series(test_df["title1_zh"])
    test_df["title2_zh"] =  chinese_clean_series(test_df["title2_zh"])

    train_df.replace('unrelated', 0, inplace=True)
    train_df.replace('agreed', 1, inplace=True)
    train_df.replace('disagreed', 2, inplace=True)

    y = list(train_df["label"])


    #単語辞書の作成

    train_t1_en = train_df["title1_en"]
    train_t2_en = train_df["title2_en"]

    test_t1_en = test_df["title1_en"]
    test_t2_en = test_df["title2_en"]

    train_t1_zh = train_df["title1_zh"]
    train_t2_zh = train_df["title2_zh"]
    test_t1_zh = test_df["title1_zh"]
    test_t2_zh = test_df["title2_zh"]

    label = train_df["label"]


    word_to_ix_en = {}
    for title1, title2 in zip(tqdm(train_t1_en), train_t2_en):
        for word in title1.split():
            if word not in word_to_ix_en.keys():
                word_to_ix_en[word] = len(word_to_ix_en)+1
        for word in title2.split():
            if word not in word_to_ix_en.keys():
                word_to_ix_en[word] = len(word_to_ix_en)+1

    for title1, title2 in zip(tqdm(test_t1_en), test_t2_en):
        for word in title1.split():
            if word not in word_to_ix_en.keys():
                word_to_ix_en[word] = len(word_to_ix_en)+1
        for word in title2.split():
            if word not in word_to_ix_en.keys():
                word_to_ix_en[word] = len(word_to_ix_en)+1


    #Chinese
    word_to_ix_zh = {}
    for title1, title2 in zip(tqdm(train_t1_zh), train_t2_zh):
        for word in title1:
            if word not in word_to_ix_zh.keys():
                word_to_ix_zh[word] = len(word_to_ix_zh)+1
        for word in title2:
            if word not in word_to_ix_zh.keys():
                word_to_ix_zh[word] = len(word_to_ix_zh)+1

    for title1, title2 in zip(tqdm(test_t1_zh), test_t2_zh):
        for word in title1:
            if word not in word_to_ix_zh.keys():
                word_to_ix_zh[word] = len(word_to_ix_zh)+1
        for word in title2:
            if word not in word_to_ix_zh.keys():
                word_to_ix_zh[word] = len(word_to_ix_zh)+1

    print("the number of english words:{}, chinese words:{}".format(len(word_to_ix_en), len(word_to_ix_zh)))

    with open('save/word_to_ix_en.pickle', mode='wb') as f:
        pickle.dump(word_to_ix_en, f)
    with open('save/word_to_ix_zh.pickle', mode='wb') as f:
        pickle.dump(word_to_ix_zh, f)
    with open('save/train_df.pickle', mode='wb') as f:
        pickle.dump(train_df, f)
    with open('save/test_df.pickle', mode='wb') as f:
        pickle.dump(test_df, f)

    print("cleaned df, word to ix saved.")


    # Aにagreeな記事たちは、Aにdisagreeな記事たちとdisagreeな関係にあるかも?

    # with open('save/word_to_ix_en.pickle', mode='rb') as f:
    #      word_to_ix_en = pickle.load(f)
    # with open('save/word_to_ix_zh.pickle', mode='rb') as f:
    #      word_to_ix_zh = pickle.load(f)
    # with open('save/train_df.pickle', mode='rb') as f:
    #      train_df = pickle.load(f)
    # with open('save/test_df.pickle', mode='rb') as f:
    #      test_df = pickle.load(f)

    #
    # title1_en = list(train_df["title1_en"])
    # title2_en = list(train_df["title2_en"])
    # title1_zh = list(train_df["title1_zh"])
    # title2_zh = list(train_df["title2_zh"])
    # labels = list(train_df["label"])
    #
    # id1 = list(train_df["tid1"])
    # id2 = list(train_df["tid2"])
    #
    # #id1_train, id1_val, train1_en, val1_en, train1_zh, val1_zh, id2_train, id2_val, train2_en, val2_en,train2_zh, val2_zh, y_train, y_val = train_test_split(id1, title1_en, title1_zh, id2, title2_en, title2_zh, labels, test_size=0.2, random_state=0)
    # training_df, val_df = train_test_split(train_df, test_size=0.2, random_state=0)
    #
    #
    # #new_data, _ = make_new_data(id1_train, id2_train, train1_en, train2_en, y_train)
    # new_data, _, _ = make_new_data(training_df)
    #
    # #print(len(new_data_en))
    #
    # train1_en, train2_en = [],[]
    # train1_zh, train2_zh = [],[]
    # y_train = []
    # for text1_en, text2_en, text1_zh, text2_zh,label in new_data:
    #         train1_en.append(text1_en)
    #         train2_en.append(text2_en)
    #         train1_zh.append(text1_zh)
    #         train2_zh.append(text2_zh)
    #         y_train.append(label)
    #
    # # new_data_zh, _ = make_new_data(id1_train, id2_train, train1_zh, train2_zh, y_train)
    # # print(len(new_data_zh))
    # # for text1, text2, label in new_data_zh:
    # #         train1_zh.append(text1)
    # #         train2_zh.append(text2)
    # #
    #
    # val1_en, val2_en = list(val_df["title1_en"]), list(val_df["title2_en"])
    # val1_zh, val2_zh = list(val_df["title1_zh"]), list(val_df["title2_zh"])
    # y_val = list(val_df["label"])
    #
    # assert len(train1_zh)==len(train1_en)  and len(y_train)==len(train1_zh)
    #
    #
    #
    # print("training data:{}, validation data:{}".format(len(y_train), len(y_val)))

    return 0

    # return (train1_en, val1_en, train1_zh, val1_zh, train2_en, val2_en,train2_zh, val2_zh, y_train, y_val)

if __name__ == "__main__":
    # train_df = pd.read_csv("data/train.csv")
    # train_df.replace('unrelated', 0, inplace=True)
    # train_df.replace('agreed', 1, inplace=True)
    # train_df.replace('disagreed', 2, inplace=True)
    _ = preprocess_()
