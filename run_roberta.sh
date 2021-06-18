export CUDA_VISIBLE_DEVICES=2
for((i=0;i<5;i++));
do   

python run_bert.py \
--model_type bert \
--model_name_or_path ./RoBERTa \
--do_test \
--data_dir ./data/data_$i \
--output_dir ./model_RoBERTa$i \
--max_seq_length 256 \
--split_num 3 \
--lstm_hidden_size 512 \
--lstm_layers 1 \
--lstm_dropout 0.1 \
--eval_steps 300 \
--per_gpu_train_batch_size 4 \
--gradient_accumulation_steps 1 \
--warmup_steps 0 \
--per_gpu_eval_batch_size 128 \
--learning_rate 5e-6 \
--adam_epsilon 1e-6 \
--weight_decay 0 \
--train_steps 5000 \
--do_train \
--do_eval
done  

python gengrate_sub_file.py