# coding: UTF-8
import torch
from tqdm import tqdm
import time
from datetime import timedelta
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import numpy as np
import pandas as pd

from models.bert import Bert
from pytorch_pretrained.optimization import BertAdam
from sklearn.metrics import accuracy_score

from pytorch_pretrained import BertTokenizer

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号
label_map = {'财经': 0, '教育': 1, '房产': 2, '娱乐': 3, '游戏': 4,
             '体育': 5, '时尚': 6, '科技': 7, '时政': 8, '家居': 9}
label_map_rev = {v: k for k, v in label_map.items()}


def build_dataset(config):
    tokenizer = BertTokenizer.from_pretrained(config.bert_path)

    def load_dataset(path, pad_size=32, type='train'):
        contents = []
        # with open(path, 'r', encoding='UTF-8') as f:
        df = pd.read_csv(path)
        if type == 'train':
            for index, rows in tqdm(df.iterrows(), desc="train data processing", total=len(list(df.iterrows()))):
                # for line in tqdm(f):
                line = rows.values[1]
                try:
                    label = label_map[rows.values[2]]
                except IndexError:
                    continue
                content = line.strip()
                if not content:
                    continue
                assert len(rows.values) == 3, 'length of rows!=3'

                for sub_content in range(0, len(content), config.pad_size - 1):  # 有一个[CLS]所以减一

                    token = tokenizer.tokenize(content[sub_content:sub_content + config.pad_size - 1])
                    token = [CLS] + token
                    seq_len = len(token)
                    mask = []
                    token_ids = tokenizer.convert_tokens_to_ids(token)

                    if pad_size:
                        if len(token) < pad_size:
                            mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                            token_ids += ([0] * (pad_size - len(token)))
                        else:
                            mask = [1] * pad_size
                            token_ids = token_ids[:pad_size]
                            seq_len = pad_size
                    contents.append((token_ids, int(label), seq_len, mask, rows.values[0]))
            return contents
        if type == 'test':
            for index, rows in tqdm(df.iterrows(), desc="test data processing", total=len(list(df.iterrows()))):
                # for line in tqdm(f):
                line = rows.values[1]
                label = -1
                content = line.strip()
                if not content:
                    continue
                assert len(rows.values) == 2, 'length of rows!=2'

                for sub_content in range(0, len(content), config.pad_size - 1):  # 有一个[CLS]所以减一

                    token = tokenizer.tokenize(content[sub_content:sub_content + config.pad_size - 1])
                    token = [CLS] + token
                    seq_len = len(token)
                    mask = []
                    token_ids = tokenizer.convert_tokens_to_ids(token)

                    if pad_size:
                        if len(token) < pad_size:
                            mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                            token_ids += ([0] * (pad_size - len(token)))
                        else:
                            mask = [1] * pad_size
                            token_ids = token_ids[:pad_size]
                            seq_len = pad_size
                    contents.append((token_ids, int(label), seq_len, mask, rows.values[0]))
            return contents

    print("**************************\npad_size : {}\n**************************".format(config.pad_size))
    # train = load_dataset(config.train_path, config.pad_size)
    train=None
    test = load_dataset(config.test_path, config.pad_size, type='test')
    return train, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = list(batches.values)
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def split_train_dev(data, dev_ratio, n_splits=1, type='Stratified'):
    '''
    随机采样
    :param data: list（tuple()）
    :param dev_ratio:  分类比例
    :return:
    '''
    # 设置随机数种子，保证每次生成的结果都是一样的
    data = pd.DataFrame(data)
    data.columns = ['tokens', 'labels', 'pad_size', 'mask', 'id']
    if type == 'Stratified':
        split = StratifiedShuffleSplit(n_splits=n_splits, test_size=dev_ratio, random_state=42)
        # # 根据labels来进行分层采样
        for train_index, test_index in split.split(data, data['labels']):
            train_set = data.loc[train_index]
            test_set = data.loc[test_index]

            yield train_set, test_set

    else:
        np.random.seed(42)
        # permutation随机生成0-len(data)随机序列
        shuffled_indices = np.random.permutation(len(data))
        # test_ratio为测试集所占的半分比
        test_set_size = int(int(len(data)) * dev_ratio)
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]
        # iloc选择参数序列中所对应的行
        return data.iloc[train_indices], data.iloc[test_indices]


def label_data():
    # 处理unlabel的数据并生成训练集
    data_path = '../data/'
    label_train_data_path = '../data/train/labeled_data.csv'
    unlabel_train_data_path = '../data/train/unlabeled_data.csv'
    test_data_path = '../data/test_data.csv'
    orgin_train_data = pd.read_csv(label_train_data_path)  # 原始的train_data
    test_data = pd.read_csv(test_data_path)  # 原始的test_data
    unlabeled_data = pd.read_csv(unlabel_train_data_path)  # 原始的unlabeled_data
    labels = ['财经', "教育", "房产", "娱乐", "游戏", "体育", "时尚", "科技", "时政", "家居"]

    # 数据抽取部分
    def juge(x):
        type_dict = {
            "财经": ["财经", "期货", "市场", "股票"],
            "房产": ["房产", "房子"],
            "家居": ["家居", "电器"],
            "教育": ["教育", "学生"],
            "科技": ["科技", "手机"],
            "时尚": ["时尚", "明星"],
            "时政": ["时政"],
            "游戏": ["游戏", "攻击", "装备"],
            "娱乐": ["主演", "演员", "综艺", "音乐", "真人秀"],
            "体育": ["体育", "锦标赛", "田径", "跳远", "世锦赛"],
        }
        keywords = []  # 遍历上面的字典，存储关键词到keywords中
        for e in type_dict:
            keywords.extend(type_dict[e])

        for type1 in type_dict:  # 对于每个类别
            for type1_sub in type_dict[type1]:  # 对于每个类别下的关键词
                if type1_sub in x:  # 如果这个词在x中出现
                    for other in keywords:  # 其他词不在x中出现
                        if other in x and (not other in type_dict[type1]):
                            return "未知"
                    return type1
        return "未知"
    # 测试这种方法在原始的train上的面的效果
    orgin_train_data['pre_labels'] = orgin_train_data['content'].apply(lambda x: juge(x))
    for label in labels:
        tmp_data = orgin_train_data[orgin_train_data["pre_labels"] == label]
        # 计算准确率
        print(label, accuracy_score(tmp_data["pre_labels"], tmp_data["class_label"]))
    # 打标数据
    unlabeled_data["class_label"] = unlabeled_data["content"].apply(lambda x: juge(x))  # 直接用class_label 忽略噪声
    unlabeled_data = unlabeled_data[unlabeled_data['class_label'] != "未知"]
    unlabeled_data = pd.concat([unlabeled_data[unlabeled_data['class_label'] == "游戏"],
                                unlabeled_data[unlabeled_data['class_label'] == "娱乐"],
                                unlabeled_data[unlabeled_data['class_label'] == "体育"]])
    print("unlabeled_data\n", unlabeled_data["class_label"].value_counts())
    # 随机截取
    game = unlabeled_data[unlabeled_data['class_label'] == "游戏"].sample(n=1000)
    tiyu = unlabeled_data[unlabeled_data['class_label'] == "体育"]
    yule = unlabeled_data[unlabeled_data['class_label'] == "娱乐"]
    unlabeled_data = pd.concat([game, tiyu, yule])

    # 合并
    all_data_train = pd.concat([unlabeled_data, orgin_train_data], axis=0, sort=False).drop(labels='pre_labels', axis=1)
    print("unlabeled_data\n", all_data_train["class_label"].value_counts())
    all_data_train.to_csv("../data/all_data_train_selected1000.csv", index=False, encoding='utf-8', header=None)


def generate_sub_file(config,test_data):
    test_data.columns = ["sub_content", "label", "len", "mask", "id"]
    test_data.drop(['sub_content', 'label', 'mask', 'len'], axis=1, inplace=True)
    print('getting npy file from '+config.npy_file_path)
    preds = np.load(config.npy_file_path)
    pre_df = pd.DataFrame(preds)
    sum_df = pd.concat([test_data, pre_df], axis=1).groupby('id').sum()
    preds = np.argmax(sum_df.values, axis=1)
    result = [label_map_rev[i] for i in list(preds)]
    sub = pd.concat([pd.read_csv(config.test_path,usecols=['id']), pd.DataFrame(result)], axis=1)
    sub.columns = ['id', 'class_label']
    sub['rank_label'] = sub['class_label'].apply(lambda x: classify_level(x))
    sub.to_csv(config.sub_path, index=False)
    print("submit file create at {}!".format(config.sub_path))

def classify_level(x):
    if x in ["财经", "时政"]:
        return "高风险"
    if x in ["房产", "科技"]:
        return "中风险"
    if x in ["教育", "时尚", "游戏"]:
        return "低风险"
    if x in ["家居", "体育", "娱乐"]:
        return "可公开"
    if x in ["未知"]:
        return "可公开"
    raise ValueError("类别不对")


def model_pre(config, train_iter_length=-1):
    # 模型加载，优化器设置
    print("loading bert model from {}".format(config.bert_path))
    model = Bert(config).to(config.device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=train_iter_length * config.num_epochs)
    return model, optimizer

if __name__ == '__main__':
    from config import config

    config.npy_file_path='../sub/bert_preds_0.85.npy'
    # build_dataset(config)
    # import numpy as np
    test_data = torch.load('../data/processed_test_data')
    test_data = pd.DataFrame(test_data)

    test_data.columns = ["sub_content", "label", "len", "mask", "id"]
    test_data.drop(['sub_content', 'label', 'mask', 'len'], axis=1, inplace=True)
    print('getting npy file from '+config.npy_file_path)
    preds = np.load(config.npy_file_path)
    pre_df = pd.DataFrame(preds)
    sum_df = pd.concat([test_data, pre_df], axis=1).groupby('id').mean()
    # np.load('../data/')
    # train_data = torch.load('../data/bert_processed_train_data')
    # trian_data, dev_data = split_train_dev(train_data, 0.2, type='Stratified')
    # label_data()

    #
    # print(len(train_set), len(test_set))
    #
    # torch.save(train_data, '../data/bert_processed_train_data')
    # torch.save(dev_data, '../data/bert_processed_dev_data')
