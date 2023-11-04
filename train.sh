#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0
python train.py \
    --model_name_or_path /home/lidailin\
    --vocab_size 21128 \
    --hidden_size 768\
    --num_hidden_layers 12\
    --num_attention_heads 12\
    --intermediate_size 3072\
    --pad_token_id 0\
    --hidden_act "gelu"\
    --hidden_dropout_prob 0.1\
    --attention_probs_dropout_prob 0.1\
    --max_position_embeddings 512\
    --type_vocab_size 2\
    --initializer_range 0.02\
    --pooler_type first_token_transform\
    --data_dir "./data"\
    --train_data_name "toutiao_train.txt"\
    --val_data_name "toutiao_val.txt"\
    --test_data_name "toutiao_test.txt"\
    --CLS_IDX "[CLS]"\
    --SEP_IDX "[SEP]"\
    --output_dir ./cache\
    --overwrite_output_dir True\
    --num_train_epochs 3\
    --use_mps_device 0\
    --per_device_train_batch_size 32\
    --warmup_steps 500\
    --weight_decay 0.01\
    --logging_dir "./logs" \
    --logging_steps 10\
    --save_steps 25000\
    --metric_for_best_model "accuracy"\
    --greater_is_better True\
    --save_total_limit 3\
    --learning_rate 5e-5\
    --vocab_path "./Myvocab/vocab.txt"
