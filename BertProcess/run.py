# coding: UTF-8
import argparse
import time
from pprint import pprint

import torch
import numpy as np
from train_eval import train, test
from utils import build_dataset, build_iterator, get_time_dif, split_train_dev, \
    generate_sub_file, model_pre
import os
import pandas as pd

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
parse = argparse.ArgumentParser()
# 需要经常修改的参数
parse.add_argument("--gpu", default='2', type=str, required=False)
parse.add_argument("--reload_data", default=False, action='store_true', help="是否重新进行数据处理")
parse.add_argument("--do_train", default=False, action='store_true')
parse.add_argument("--do_test", default=False, action='store_true')
# 文件保存、加载路径
parse.add_argument("--train_path", default='../data/all_data_train_selected1000_加规则.csv', type=str,
                   help="original train data witch contents all labels")
parse.add_argument("--sub_path", default='../sub/bert_submit.csv', type=str, help="submit file path")
parse.add_argument("--npy_file_path", default='../sub/bert_preds.npy', type=str, help="npy_file path")
parse.add_argument("--test_path", default='../data/test_data.csv', type=str, help="test data path")
parse.add_argument("--save_path", default='../results/bert_model_b46_p128.ckpt', type=str, help="model save path")
parse.add_argument("--bert_path", default='./bert-base', type=str, help="bert path")
# 改变会有提升的
parse.add_argument("--batch_size", default=46, type=int)
parse.add_argument("--pad_size", default=128, type=int, help="sentence maxlength")
parse.add_argument("--n_split", default=1, type=int)
# 一般固定的参数
parse.add_argument("--num_classes", default=10, type=int)
parse.add_argument("--iter_print", default=200, type=int, help='迭代多少次打印出dev上的效果')
parse.add_argument("--hidden_size", default=768, type=int)
parse.add_argument("--device", default="cuda", type=str)
parse.add_argument("--require_improvement", default=300, type=int, help="若超过1000batch效果还没提升，则提前结束训练")
parse.add_argument("--learning_rate", default=5e-5, type=float)
parse.add_argument("--num_epochs", default=100, type=int)
parse.add_argument("--model", default='bert', type=str, help='bert or ERNIE')

config = parse.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu


def set_seed(seed=2018):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样


if __name__ == '__main__':
    # config = Config()

    config.processed_data_path = "../data/bert_processed_split_windows_{}_data_{}"
    # config.processed_data_path = "../data/{}_data_p{}"
    config.processed_data_path = config.train_path.split('.')[0] + '_processed_{}_data_p{}'

    print("model:{}\npad_size:{}\nbatch_size:{}\ndevice:{}\nsave_path:{}\n".format(config.model, config.pad_size,
                                                                                   config.batch_size, config.device,
                                                                                   config.save_path))
    pprint(config)
    if os.path.exists(config.processed_data_path.format('train', config.pad_size)) and config.reload_data == False:
        print("Loading data...")
        train_data = torch.load(config.processed_data_path.format('train', config.pad_size))
        test_data = torch.load(config.processed_data_path.format('test', config.pad_size))
    else:
        print("processing data...")
        train_data, test_data = build_dataset(config)
        print("saving data...")
    # torch.save(train_data, config.processed_data_path.format('train', config.pad_size))
    torch.save(test_data, config.processed_data_path.format('test', config.pad_size))
    print('train_data\'s length: {} \ntest_data\'s length: {}'.format(len(train_data), len(test_data)))
    test_data = pd.DataFrame(test_data)
    test_iter = build_iterator(test_data, config)

    if config.do_train:
        # 训练
        flag = 0
        preds = None
        for train_data_split, dev_data_split in tqdm(
                split_train_dev(train_data, dev_ratio=0.2, n_splits=config.n_split),
                desc="n fold", total=config.n_split):
            start_time = time.time()
            train_iter = build_iterator(train_data_split, config)
            dev_iter = build_iterator(dev_data_split, config)
            model, optimizer = model_pre(config, len(train_iter))
            train(config, model, train_iter, dev_iter, optimizer)
            time_dif = get_time_dif(start_time)
            print("Time usage:", time_dif)
            if flag == 0:
                flag += 1
                preds = test(config, model, test_iter)
            else:
                preds += test(config, model, test_iter)
        np.save(config.npy_file_path, preds)
        print(config.npy_file_path+'saved')

    if config.do_test:
        model, optimizer = model_pre(config)
        preds = test(config, model, test_iter)
        np.save(config.npy_file_path, preds)
        print(config.npy_file_path+'saved')

    generate_sub_file(config, test_data)
