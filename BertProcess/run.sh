
# train
python run.py \
--do_train --gpu 0 \
--batch_size 48 \
--n_split 1 \
--save_path ../results/bert_model_b46_p128.ckpt \
--train_path ../data/all_data_train_selected1000.csv \
--do_train 

python run.py \
--do_test \
--batch_size 512 \
