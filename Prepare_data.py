import os
import re
import math
import torch
import random
import datetime
import pandas as pd


class EntityDictionary:
    '''
    实体字典，讲实体映射到idx
    '''
    def __init__(self):
        self.idx2entity = []
        self.entity2idx = {}

    def add_entity(self, e):
        if e not in self.entity2idx:
            self.entity2idx[e] = len(self.idx2entity)
            self.idx2entity.append(e)

    def __len__(self):
        return len(self.idx2entity)


class DataLoader:
    '''
    dataloader，加载user, item, feature(包括：电影的tag， 导演， 豆瓣分类， 日期）, 用户的评论，rating
    '''
    def __init__(self, data_path, tokenizer, seq_len):
        self.user = EntityDictionary()
        self.item = EntityDictionary()
        self.max_rating = float('-inf')
        self.min_rating = float('inf')
        self.initialize(data_path)
        self.feature_set = set()    #feature set 存储所出现过的所有feature
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.train, self.valid, self.test = self.load_data(data_path)

    def initialize(self, data_path):
        '''
        初始化，从数据集中读取数据，user, item加入到entity_dictionary中，记录最大，最小rating
        '''
        assert os.path.exists(data_path)
        reviews = pd.read_csv(data_path)
        for i in range(reviews.shape[0]):
            #user, item加入到entity_dictionary中
            self.user.add_entity(reviews['USER_MD5'][i])
            self.item.add_entity(reviews['MOVIE_ID'][i])
            #处理rating
            rating = reviews['RATING'][i]
            if self.max_rating < rating:
                self.max_rating = rating
            if self.min_rating > rating:
                self.min_rating = rating

    def load_data(self, data_path):
        '''
        加载数据，处理方法：电影的tag， 导演， 豆瓣分类， 日期连接成为feature， 每条数据以字典格式存储
        user: , item:, rating:, text: , feature:
        随机划分数据集， train:test:validation=8:1:1

        '''
        data = []
        reviews = pd.read_csv(data_path)
        for i in range(reviews.shape[0]):
            feature = []
            # feature中加入tags
            tags = reviews['TAGS'][i].split('/')
            for j in range(3):
                feature.append(tags[j])
                # feature_set 的结构set，存储当前数据集出现过的feature
                self.feature_set.add(tags[j])

            # feature中加入genres
            genres = reviews['GENRES'][i].split('/')
            feature.append(genres[0])
            self.feature_set.add(genres[0])

            # feature中加入director
            director = reviews['DIRECTORS'][i].split('/')
            feature.append(director[0])
            self.feature_set.add(director[0])

            # 加入year--need to consider again
            year = int(reviews['YEAR'][i])
            feature.append(str(year))
            self.feature_set.add(str(year))

            # 加入评论
            comment = str(reviews['SELECTED'][i])
            # 使用bert_tokenize
            tokens = self.tokenizer(comment)['input_ids']
            text = self.tokenizer.decode(tokens[:self.seq_len])  # keep seq_len tokens at most
            data.append({'user': self.user.entity2idx[reviews['USER_MD5'][i]],
                         'item': self.item.entity2idx[reviews['MOVIE_ID'][i]],
                         'rating': reviews['RATING'][i],
                         'text': text,
                         'feature': feature})

        # 划分数据集，train:test:validation=8:1:1
        train_num = math.ceil(reviews.shape[0] * 0.8)
        test_num = math.ceil(reviews.shape[0] * 0.1)
        train_idx = set(random.sample(range(reviews.shape[0]), train_num))
        left_reviews = set(range(reviews.shape[0])) - train_idx
        test_idx = set(random.sample(list(left_reviews), test_num))
        vali_idx = left_reviews - test_idx
        train_idx = list(train_idx)
        test_idx = list(test_idx)
        vali_idx = list(vali_idx)
        # 读入数据
        train, valid, test = [], [], []
        for idx in train_idx:
            train.append(data[idx])
        for idx in vali_idx:
            valid.append(data[idx])
        for idx in test_idx:
            test.append(data[idx])
        return train, valid, test


class Batchify:
    def __init__(self, data, tokenizer, bos, eos, seq_len, batch_size=128, shuffle=False):
        '''
        初始化，将(user, item, feature)拼接成为src_prompt，src_prompt使用tokenizer成为prompt准备使用。
        并在评论的首尾分别加入<bos>, <eos>方便之后的explanation generation
        Args:
            bos: 开始标志
            eos:结束标签
            seq_len: prompt 的指定长度
            shuffle:是否shuffle, default = false
        '''
        text, self.feature, src_prompt, r = [], [], [], []  #分别存储文本， 特征， prompt(user,item, feature), rating
        user, item = [], []
        for review in data:
            r.append(review['rating'])
            temp = []   #temp中将顺序加入user, item feature, 以拼接成为src_prompt
            temp.append(str(review['user']))
            temp.append(str(review['item']))
            for f in review['feature']:
                temp.append(f)
                self.feature.append(f)
            src_prompt.append(' '.join(temp))
            text.append('{} {} {}'.format(bos, review['text'], eos))    #评论的首尾分别加入<bos>, <eos>
            user.append(review['user'])
            item.append(review['item'])


        encoded_inputs = tokenizer(text, padding=True, return_tensors='pt')
        self.seq = encoded_inputs['input_ids'].contiguous()
        self.mask = encoded_inputs['attention_mask'].contiguous()   #获取mask
        encoded_features = tokenizer(src_prompt, padding=True, return_tensors='pt')
        self.prompt = encoded_features['input_ids'][:, :seq_len].contiguous()
        self.user = user
        self.item = item
        self.rating = torch.tensor(r, dtype=torch.float).contiguous()
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.sample_num = len(data)
        self.index_list = list(range(self.sample_num))
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0

    def next_batch(self):
        if self.step == self.total_step:
            self.step = 0
            if self.shuffle:
                random.shuffle(self.index_list)

        start = self.step * self.batch_size
        offset = min(start + self.batch_size, self.sample_num)
        self.step += 1
        index = self.index_list[start:offset]
        seq = self.seq[index]  # (batch_size, seq_len)
        mask = self.mask[index]
        prompt = self.prompt[index]
        rating = self.rating[index]
        return seq, mask, prompt, rating

def now_time():
    return '[' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + ']: '


def postprocessing(string):
    '''
    具体见：https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    使句子的标点分割更符合现实应用场景
    '''
    string = re.sub('\'s', ' \'s', string)
    string = re.sub('\'m', ' \'m', string)
    string = re.sub('\'ve', ' \'ve', string)
    string = re.sub('n\'t', ' n\'t', string)
    string = re.sub('\'re', ' \'re', string)
    string = re.sub('\'d', ' \'d', string)
    string = re.sub('\'ll', ' \'ll', string)
    string = re.sub('\(', ' ( ', string)
    string = re.sub('\)', ' ) ', string)
    string = re.sub(',+', ' , ', string)
    string = re.sub(':+', ' , ', string)
    string = re.sub(';+', ' . ', string)
    string = re.sub('\.+', ' . ', string)
    string = re.sub('!+', ' ! ', string)
    string = re.sub('\?+', ' ? ', string)
    string = re.sub(' +', ' ', string).strip()
    return string


def ids2tokens(ids, tokenizer, eos):
    '''
    将获得的ids映射回token
    '''
    text = tokenizer.decode(ids)
    text = postprocessing(text)
    tokens = []
    for token in text.split():
        if token == eos:
            #检测到结束符，停止
            break
        tokens.append(token)
    return tokens

class Batchify_for_comp:
    '''
    原文模型所需要用的bachify
    '''
    def __init__(self, data, tokenizer, bos, eos, batch_size=128, shuffle=False):
        user, item, rating, exp, self.feature = [], [], [], [], []
        for review in data:
            user.append(review['user'])
            item.append(review['item'])
            rating.append(review['rating'])
            exp.append('{} {} {}'.format(bos, review['text'], eos)) #在explanation前后分别加上<bos><eos>
            self.feature.append(review['feature'])


        #分别初始化 explain user item rating
        encoded_inputs = tokenizer(exp, padding=True, return_tensors='pt')
        self.explain = encoded_inputs['input_ids'].contiguous()
        self.mask = encoded_inputs['attention_mask'].contiguous()
        self.user = torch.tensor(user, dtype=torch.int64).contiguous()
        self.item = torch.tensor(item, dtype=torch.int64).contiguous()
        self.rating = torch.tensor(rating, dtype=torch.float).contiguous()
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.sample_num = len(data)
        self.index_list = list(range(self.sample_num))
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0

    def next_batch(self):
        if self.step == self.total_step:
            self.step = 0
            if self.shuffle:
                random.shuffle(self.index_list)

        start = self.step * self.batch_size
        offset = min(start + self.batch_size, self.sample_num)
        self.step += 1
        index = self.index_list[start:offset]
        user = self.user[index]
        item = self.item[index]
        rating = self.rating[index]
        explain = self.explain[index]
        mask = self.mask[index]
        return user, item, rating, explain, mask