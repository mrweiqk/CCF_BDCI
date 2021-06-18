import os

import pandas as pd
import numpy as np
import argparse

label_map = {'财经': 0, '教育': 1, '房产': 2, '娱乐': 3, '游戏': 4,
             '体育': 5, '时尚': 6, '科技': 7, '时政': 8, '家居': 9}
label_map_rev = {v: k for k, v in label_map.items()}

parser = argparse.ArgumentParser()
parser.add_argument("--model_prefix", default='model_bert_split5_', type=str)
parser.add_argument("--out_path", default='./sub/sub.csv', type=str)
args = parser.parse_args()


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
    raise ValueError("类别不对{}".format(x))


k = 5
flag = 1
np_sum = None
for i in range(k):
    if os.path.exists(f'./{args.model_prefix}{i}/logits.npy'):
        if flag:
            np_sum = np.load(f'./{args.model_prefix}{i}/logits.npy')
        else:
            np_sum += np.load(f'./{args.model_prefix}{i}/logits.npy')

labels = np.argmax(np_sum, axis=1)
result = [label_map_rev[i] for i in list(labels)]
test_data = pd.read_csv('./data/test_data.csv', usecols=['id'])
sub = pd.concat([test_data, pd.DataFrame(result)], axis=1)
sub.columns = ['id', 'class_label']
sub['rank_label'] = sub['class_label'].apply(lambda x: classify_level(x))
sub.to_csv(args.out_path, index=False)

np_sum = np_sum / k
np.save('./sub/np_sum.npy',np_sum)