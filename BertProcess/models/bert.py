# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained import BertModel, BertTokenizer
import os

# class Config(object):
#
#     """配置参数"""
#     def __init__(self):
#
#         self.reload_data = False
#         self.pad_size = 128  # 每句话处理成的长度(短填长切)
#         self.processed_data_path = "../data/bert_processed_{}_data_{}"
#
#         self.model = os.path.basename(__file__).split('.')[0]
#         self.train_path = '../data/all_data_train_selected1000.csv'
#         # self.dev_path = '../data/processed_train_data.csv'
#         self.test_path = '../data/test_data.csv'
#         # label_map = {'财经': 0, '教育': 1, '房产': 2, '娱乐': 3, '游戏': 4,
#         #              '体育': 5, '时尚': 6, '科技': 7, '时政': 8, '家居': 9}
#         self.class_list = ['财经',"教育","房产","娱乐","游戏","体育","时尚","科技","时政","家居"]
#                         # 类别名单
#         self.batch_size = 46                                           # mini-batch大小
#
#
#         self.save_path =  '../results/{}_{}_{}.ckpt'.format(self.model,self.pad_size,self.batch_size)        # 模型训练结果
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
#
#         self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
#         self.num_classes = len(self.class_list)                         # 类别数
#         self.num_epochs = 100                                             # epoch数
#         self.learning_rate = 5e-5                                       # 学习率
#         self.bert_path = './bert_pretrain'
#         self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
#         self.hidden_size = 768

class Bert(nn.Module):

    def __init__(self, config):
        super(Bert, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out


# if __name__ == '__main__':
#     con = Config()