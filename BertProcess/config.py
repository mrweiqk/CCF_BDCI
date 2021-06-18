import argparse
import torch
import os

parse = argparse.ArgumentParser()
parse.add_argument("--gpu", default='2', type=str, required=False)
parse.add_argument("--reload_data", default=False, type=bool, help="是否重新进行数据处理")
parse.add_argument("--pad_size", default=128, type=int, help="sentence maxlength")
parse.add_argument("--model", default='bert', type=str, help='bert or ERNIE')
parse.add_argument("--train_path", default='../data/all_data_train_selected1000.csv', type=str,
                   help="original train data witch contents all labels")
parse.add_argument("--test_path", default='../data/test_data.csv', type=str, help="test data path")
parse.add_argument("--batch_size", default=46, type=int)
parse.add_argument("--device", default="cuda", type=str)
parse.add_argument("--require_improvement", default=300, type=int, help="若超过1000batch效果还没提升，则提前结束训练")
parse.add_argument("--num_classes", default=10, type=int)
parse.add_argument("--num_epochs", default=100, type=int)
parse.add_argument("--learning_rate", default=5e-5, type=float)
parse.add_argument("--bert_path", default='./bert_pretrain', type=str, help="bert path")
parse.add_argument("--hidden_size", default=768, type=int)
parse.add_argument("--save_path", default='../results/bert_model_b46_p128.ckpt', type=str, help="model save path")
parse.add_argument("--do_train", default=True, action='store_true')
parse.add_argument("--do_test", default=True, action='store_true')
parse.add_argument("--n_split", default=1, type=int)

config = parse.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu