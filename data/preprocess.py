import pandas as pd
import os
import random

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
from sklearn.metrics import accuracy_score
orgin_train_data['pre_labels'] = orgin_train_data['content'].apply(lambda x:juge(x))
orgin_train_data.head()
for label in labels:
    tmp_data = orgin_train_data[orgin_train_data["pre_labels"]==label]
    # 计算准确率
    print(label,accuracy_score(tmp_data["pre_labels"],tmp_data["class_label"]))
# 打标数据
unlabeled_data["class_label"] = unlabeled_data["content"].apply(lambda x:juge(x)) # 直接用class_label 忽略噪声
unlabeled_data = unlabeled_data[unlabeled_data['class_label']!="未知"]
unlabeled_data = pd.concat([unlabeled_data[unlabeled_data['class_label']=="游戏"],
                            unlabeled_data[unlabeled_data['class_label']=="娱乐"],
                            unlabeled_data[unlabeled_data['class_label']=="体育"]])
unlabeled_data["class_label"].value_counts()

# 随机截取
game = unlabeled_data[unlabeled_data['class_label']=="游戏"].sample(n=1000)
tiyu = unlabeled_data[unlabeled_data['class_label']=="体育"]
yule = unlabeled_data[unlabeled_data['class_label']=="娱乐"].sample(n=1000)
unlabeled_data = pd.concat([game,tiyu,yule])
# 保存数据
all_data_train = pd.concat([unlabeled_data,orgin_train_data],axis=0).drop(labels='pre_labels',axis=1)
print(all_data_train["class_label"].value_counts())
all_data_train.to_csv("../data/all_data_train_selected1000_加规则.csv",index=False,encoding='utf-8',header=None)




# 生成交叉验证的数据
label_map = {'财经': 0, '教育': 1, '房产': 2, '娱乐': 3, '游戏': 4,
             '体育': 5, '时尚': 6, '科技': 7, '时政': 8, '家居': 9}
train_df = pd.read_csv("./all_data_train_selected1000_加规则.csv",names=['id','content','label'])
test_df = pd.read_csv("./test_data.csv")
train_df['label'] = train_df['label'].apply(lambda x:label_map[x])
train_df['label'] = train_df['label'].astype(int)
test_df['label'] = -1
index = set(range(train_df.shape[0]))
K_fold = []
for i in range(5):
    if i == 4:
        tmp = index
    else:
        tmp = random.sample(index, int(1.0 / 5 * train_df.shape[0]))
    index = index - set(tmp)
    print("Number:", len(tmp))
    K_fold.append(tmp)
for i in range(5):
    print("Fold", i)
    os.system("mkdir data_{}".format(i))
    dev_index = list(K_fold[i])
    train_index = []
    for j in range(5):
        if j != i:
            train_index += K_fold[j]
    train_df.iloc[train_index].to_csv("data_{}/train.csv".format(i))
    train_df.iloc[dev_index].to_csv("data_{}/dev.csv".format(i))
    test_df.to_csv("data_{}/test.csv".format(i))





