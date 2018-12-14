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

class BERT_Classifier(nn.Module):
    def __init__(self, bert_model, embedding_dim=768, target_size=3):
        super(BERT_Classifier, self).__init__()

        self.embedding_dim=embedding_dim

        self.bert_model = bert_model

        self.fc1 = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.fc1_drop = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(self.embedding_dim, target_size)

    def forward(self, input_ids, input_mask):
        batch = len(input_ids)

        last_encoder_layer, _ = self.bert_model(input_ids, token_type_ids=None, attention_mask=input_mask, output_all_encoded_layers=False)

        first_token_tensor = last_encoder_layer[:, 0]

        fc1 = self.fc1_drop(F.relu(self.fc1(first_token_tensor)))
        fc2 = self.fc2(fc1)

        return fc2




class LSTM_Classifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size_en, vocab_size_zh, target_size=3, seq_length_en=50, seq_length_zh=140):
        super(LSTM_Classifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.word_embeddings_en = nn.Embedding(vocab_size_en+1, embedding_dim, padding_idx=0)
        self.word_embeddings_zh = nn.Embedding(vocab_size_zh+1, embedding_dim, padding_idx=0)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm_en = nn.LSTM(embedding_dim, hidden_dim, batch_first=False, num_layers=2)
        self.lstm_zh = nn.LSTM(embedding_dim, hidden_dim, batch_first=False, num_layers=2)

        # The linear layer that maps from hidden state space to tag space
        self.fc1 = nn.Linear(hidden_dim*2, hidden_dim*2)
        self.fc1_drop = nn.Dropout(p=0.5, inplace=False)

        self.fc2 = nn.Linear(hidden_dim*2, target_size)
        self.initial_hidden = self.init_hidden()


        self.seq_length_en=seq_length_en
        self.seq_length_zh=seq_length_zh

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, title1_en, title2_en, title1_zh, title2_zh):
        batch = title1_en.shape[0]

        embeds1_en = self.word_embeddings_en(title1_en)
        embeds2_en = self.word_embeddings_en(title2_en)

        embeds1_zh = self.word_embeddings_zh(title1_zh)
        embeds2_zh = self.word_embeddings_zh(title2_zh)

        # seq_length * batch * feature_dims
        embeds1_en = embeds1_en.view(self.seq_length_en, batch, self.embedding_dim)
        embeds2_en = embeds2_en.view(self.seq_length_en, batch, self.embedding_dim)

        embeds1_zh = embeds1_zh.view(self.seq_length_zh, batch, self.embedding_dim)
        embeds2_zh = embeds2_zh.view(self.seq_length_zh, batch, self.embedding_dim)

        #print("embeds1_en", embeds1_en.size())

        lstm_out1_en, self.hidden = self.lstm_en(embeds1_en)#, self.initial_hidden)
        lstm_out2_en, self.hidden = self.lstm_en(embeds2_en)
        lstm_out1_zh, self.hidden = self.lstm_zh(embeds1_zh)
        lstm_out2_zh, self.hidden = self.lstm_zh(embeds1_zh)

        en_sum = lstm_out1_en[-1] + lstm_out2_en[-1]
        zh_sum = lstm_out1_zh[-1] + lstm_out2_zh[-1]
        #print("embedding size:",en_sum.size(), zh_sum.size())

        concat = torch.cat((en_sum, zh_sum), dim=1)
        #print("lstm out:", lstm_out1[-1].size())
        #print("concat:", concat.size())

        fc1 = self.fc1_drop(F.relu(self.fc1(concat)))
        fc2 = self.fc2(fc1)

        return fc2



class MLP_Classifier(nn.Module):

    def __init__(self, embedding_dim, vocab_size, target_size=3, seq_length=50):
        super(MLP_Classifier, self).__init__()
        self.embedding_dim = embedding_dim
        self.word_embeddings = nn.Embedding(vocab_size+1, embedding_dim, padding_idx=0)


        # The linear layer that maps from hidden state space to tag space
        self.fc1 = nn.Linear(embedding_dim*2, embedding_dim*2)
        self.fc1_bn = nn.BatchNorm1d(embedding_dim*2)
        self.fc1_drop = nn.Dropout(p=0.5, inplace=False)

        self.fc2 = nn.Linear(embedding_dim*2, target_size)

        self.seq_length=seq_length

    def forward(self, sentence1, sentence2):
        embeds1 = self.word_embeddings(sentence1)
        embeds1 = torch.sum(embeds1, 1)
        #print("embed", embeds1.size())


        embeds2 = self.word_embeddings(sentence2)
        embeds2 = torch.sum(embeds2, 1)

        #print("embedding size:",embeds1.size(), len(sentence1))

        #embeds1 = embeds1.view(self.seq_length, len(sentence1), self.embedding_dim)
        #embeds2 = embeds2.view(self.seq_length, len(sentence1), self.embedding_dim)

        concat = torch.cat((embeds1, embeds2), dim=1)
        #print("concat:", concat.size())

        fc1 = self.fc1_drop(F.relu(self.fc1_bn(self.fc1(concat))))
        fc2 = self.fc2(fc1)

        return fc2

#Combine English and Chinese.

class Twolang_Classifier(nn.Module):

    def __init__(self, embedding_dim, vocab_size_en, vocab_size_zh, target_size=3, seq_length_en=50, seq_length_zh=100, kernel_num=64):
        super(Twolang_Classifier, self).__init__()

        self.embedding_dim = embedding_dim
        self.seq_length_en=seq_length_en
        self.seq_length_zh=seq_length_zh

        self.word_embeddings_en = nn.Embedding(vocab_size_en+1, embedding_dim, padding_idx=0)
        self.word_embeddings_zh = nn.Embedding(vocab_size_zh+1, embedding_dim, padding_idx=0)


        self.kernel_num=kernel_num
        self.conv2_en = nn.Conv2d(1, kernel_num, (2, embedding_dim))
        self.conv3_en = nn.Conv2d(1, kernel_num, (3, embedding_dim))
        self.conv4_en = nn.Conv2d(1, kernel_num, (4, embedding_dim))

        self.conv2 = nn.Conv2d(1, kernel_num, (2, embedding_dim))
        self.conv3 = nn.Conv2d(1, kernel_num, (3, embedding_dim))
        self.conv4 = nn.Conv2d(1, kernel_num, (4, embedding_dim))
        #self.conv5 = nn.Conv2d(1, kernel_num, (5, embedding_dim))

        self.Max2_pool_en = nn.MaxPool2d((self.seq_length_en-2+1, 1))
        self.Max3_pool_en = nn.MaxPool2d((self.seq_length_en-3+1, 1))
        self.Max4_pool_en = nn.MaxPool2d((self.seq_length_en-4+1, 1))
        #self.Max5_pool = nn.MaxPool2d((self.seq_length-5+1, 1))
        self.Max2_pool = nn.MaxPool2d((self.seq_length_zh-2+1, 1))
        self.Max3_pool = nn.MaxPool2d((self.seq_length_zh-3+1, 1))
        self.Max4_pool = nn.MaxPool2d((self.seq_length_zh-4+1, 1))


        # The linear layer that maps from hidden state space to tag space
        #self.fc1 = nn.Linear(embedding_dim*4, embedding_dim*4)
        #self.fc1_bn = nn.BatchNorm1d(embedding_dim*4)
        # self.fc1 = nn.Linear(embedding_dim+kernel_num*3, embedding_dim+kernel_num*3)
        self.fc1 = nn.Linear(kernel_num*6, kernel_num*6)

        self.fc1_bn = nn.BatchNorm1d(kernel_num*6)
        self.fc1_drop = nn.Dropout(p=0.5, inplace=False)

        self.fc2 = nn.Linear(kernel_num*6, target_size)


    def forward(self, title1_en, title2_en, title1_zh, title2_zh):
        batch = title1_en.shape[0]

        embeds1_en = self.word_embeddings_en(title1_en)
        #embeds1_en = torch.sum(embeds1_en, 1)
        embeds1_en = embeds1_en.view(batch, 1, self.seq_length_en, self.embedding_dim)

        embeds2_en = self.word_embeddings_en(title2_en)
        #embeds2_en = torch.sum(embeds2_en, 1)
        embeds2_en = embeds2_en.view(batch, 1, self.seq_length_en, self.embedding_dim)

        #Convolution
        embeds1_x2 = F.relu(self.conv2_en(embeds1_en))
        embeds1_x3 = F.relu(self.conv3_en(embeds1_en))
        embeds1_x4 = F.relu(self.conv4_en(embeds1_en))
        #embeds1_x5 = F.relu(self.conv5(embeds1_zh))

        embeds2_x2 = F.relu(self.conv2_en(embeds2_en))
        embeds2_x3 = F.relu(self.conv3_en(embeds2_en))
        embeds2_x4 = F.relu(self.conv4_en(embeds2_en))
        #embeds2_x5 = F.relu(self.conv5(embeds2_zh))

        # Pooling
        embeds1_x2 = self.Max2_pool_en(embeds1_x2).view(batch, -1)
        embeds1_x3 = self.Max3_pool_en(embeds1_x3).view(batch, -1)
        embeds1_x4 = self.Max4_pool_en(embeds1_x4).view(batch, -1)
        #embeds1_x5 = self.Max5_pool(embeds1_x5).view(batch, -1)

        embeds2_x2 = self.Max2_pool_en(embeds2_x2).view(batch, -1)
        embeds2_x3 = self.Max3_pool_en(embeds2_x3).view(batch, -1)
        embeds2_x4 = self.Max4_pool_en(embeds2_x4).view(batch, -1)
        #embeds2_x5 = self.Max5_pool(embeds2_x5).view(batch, -1)


        embeds1_en = torch.cat((embeds1_x2, embeds1_x3, embeds1_x4), dim=1)
        embeds2_en = torch.cat((embeds2_x2, embeds2_x3, embeds2_x4), dim=1)


        en_sum = embeds1_en + embeds2_en



        embeds1_zh = self.word_embeddings_zh(title1_zh)
        #embeds1_zh = torch.sum(embeds1_zh, 1)
        #For CNN.
        embeds1_zh = embeds1_zh.view(batch, 1, self.seq_length_zh, self.embedding_dim)

        embeds2_zh = self.word_embeddings_zh(title2_zh)
        #embeds2_zh = torch.sum(embeds2_zh, 1)
        #For CNN.
        embeds2_zh = embeds2_zh.view(batch, 1, self.seq_length_zh, self.embedding_dim)

        #Convolution
        embeds1_x2 = F.relu(self.conv2(embeds1_zh))
        embeds1_x3 = F.relu(self.conv3(embeds1_zh))
        embeds1_x4 = F.relu(self.conv4(embeds1_zh))
        #embeds1_x5 = F.relu(self.conv5(embeds1_zh))

        embeds2_x2 = F.relu(self.conv2(embeds2_zh))
        embeds2_x3 = F.relu(self.conv3(embeds2_zh))
        embeds2_x4 = F.relu(self.conv4(embeds2_zh))
        #embeds2_x5 = F.relu(self.conv5(embeds2_zh))

        # Pooling
        embeds1_x2 = self.Max2_pool(embeds1_x2).view(batch, -1)
        embeds1_x3 = self.Max3_pool(embeds1_x3).view(batch, -1)
        embeds1_x4 = self.Max4_pool(embeds1_x4).view(batch, -1)
        #embeds1_x5 = self.Max5_pool(embeds1_x5).view(batch, -1)

        embeds2_x2 = self.Max2_pool(embeds2_x2).view(batch, -1)
        embeds2_x3 = self.Max3_pool(embeds2_x3).view(batch, -1)
        embeds2_x4 = self.Max4_pool(embeds2_x4).view(batch, -1)
        #embeds2_x5 = self.Max5_pool(embeds2_x5).view(batch, -1)


        embeds1_zh = torch.cat((embeds1_x2, embeds1_x3, embeds1_x4), dim=1)
        embeds2_zh = torch.cat((embeds2_x2, embeds2_x3, embeds2_x4), dim=1)

        zh_sum = embeds1_zh + embeds2_zh

        #print("embedding size:",embeds1.size(), len(sentence1))

        #embeds1 = embeds1.view(self.seq_length, len(sentence1), self.embedding_dim)
        #embeds2 = embeds2.view(self.seq_length, len(sentence1), self.embedding_dim)

        #concat = torch.cat((embeds1_en, embeds2_en, embeds1_zh, embeds2_zh), dim=1)
        concat = torch.cat((en_sum, zh_sum), dim=1)

        fc1 = self.fc1_drop(F.relu(self.fc1_bn(self.fc1(concat))))
        fc2 = self.fc2(fc1)

        return fc2


class Text_CNN_Classifier(nn.Module):

    def __init__(self, embedding_dim, vocab_size, target_size=3, seq_length=50):
        super(Text_CNN_Classifier, self).__init__()
        self.embedding_dim = embedding_dim
        self.word_embeddings = nn.Embedding(vocab_size+1, embedding_dim, padding_idx=0)
        self.seq_length=seq_length

        self.conv3_1 = nn.Conv2d(1, 1, (3, embedding_dim))
        self.conv4_1 = nn.Conv2d(1, 1, (4, embedding_dim))
        self.conv5_1 = nn.Conv2d(1, 3, (5, embedding_dim))
        self.conv3_2 = nn.Conv2d(1, 1, (3, embedding_dim))
        self.conv4_2 = nn.Conv2d(1, 1, (4, embedding_dim))
        self.conv5_2 = nn.Conv2d(1, 1, (5, embedding_dim))

        self.Max3_pool = nn.MaxPool2d((self.seq_length-3+1, 1))
        self.Max4_pool = nn.MaxPool2d((self.seq_length-4+1, 1))
        self.Max5_pool = nn.MaxPool2d((self.seq_length-5+1, 1))

        # The linear layer that maps from hidden state space to tag space
        self.fc1 = nn.Linear(6, target_size)


    def forward(self, sentence1, sentence2):
        batch = len(sentence1)
        embeds1 = self.word_embeddings(sentence1)
        embeds2 = self.word_embeddings(sentence2)

        embeds1 = embeds1.view(len(sentence1), 1, self.seq_length, self.embedding_dim)
        embeds2 = embeds2.view(len(sentence2), 1, self.seq_length, self.embedding_dim)

        # Convolution
        embeds1_x1 = F.relu(self.conv3_1(embeds1))
        embeds1_x2 = F.relu(self.conv4_1(embeds1))
        embeds1_x3 = F.relu(self.conv5_1(embeds1))
    #         embeds2_x1 = F.relu(self.conv3_2(embeds2))
    #         embeds2_x2 = F.relu(self.conv4_2(embeds2))
    #         embeds2_x3 = F.relu(self.conv5_2(embeds2))
        embeds2_x1 = F.relu(self.conv3_1(embeds2))
        embeds2_x2 = F.relu(self.conv4_1(embeds2))
        embeds2_x3 = F.relu(self.conv5_1(embeds2))

        # Pooling
        embeds1_x1 = self.Max3_pool(embeds1_x1)
        embeds1_x2 = self.Max4_pool(embeds1_x2)
        embeds1_x3 = self.Max5_pool(embeds1_x3)
        embeds2_x1 = self.Max3_pool(embeds2_x1)
        embeds2_x2 = self.Max4_pool(embeds2_x2)
        embeds2_x3 = self.Max5_pool(embeds2_x3)

        #print("max pool size:", embeds2_x3.size())

        concat = torch.cat((embeds1_x1, embeds1_x2, embeds1_x3, embeds2_x1, embeds2_x2, embeds2_x3), -1)
        x = concat.view(batch, -1)
        #print("concat:", x.size())

        fc1 = self.fc1(x)
        #print("fc1:", fc1.size())

        return fc1
