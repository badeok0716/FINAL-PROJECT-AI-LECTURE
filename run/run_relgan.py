# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : run_relgan.py
# @Time         : Created at 2019-05-28
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import sys
from subprocess import call

import os
import argparse

parser = argparse.ArgumentParser()
# aug exp
parser.add_argument('--key', default='aug_exp', type=str)
parser.add_argument('--train_ratio', type=float)
parser.add_argument('--trial', type=int)
parser.add_argument('--aug', type=str)
parser.add_argument('--aug_place', default='all', type=str)
args = parser.parse_args()

# Job id and gpu_id
job_id, gpu_id = 1,0

# Executables
executable = 'python'  # specify your own python interpreter path here
rootdir = '../'
scriptname = 'main.py'

# ===Program===
if_test = int(False)
run_model = 'relgan'
CUDA = int(True)
oracle_pretrain = int(True)
gen_pretrain = int(False)
dis_pretrain = int(False)
MLE_train_epoch = 150
ADV_train_epoch = 3000
tips = 'RelGAN experiments'

# ===Oracle or Real===
if_real_data = [int(False), int(True), int(True)]
dataset = ['oracle', 'image_coco', 'emnlp_news']
loss_type = 'rsgan'
vocab_size = [5000, 0, 0]
temp_adpt = 'exp'
temperature = [1, 100, 100]

# aug
key = args.key #'aug_exp'
train_ratio = args.train_ratio #0.1
trial = args.trial #0
aug = args.aug #'noaug'
aug_place = args.aug_place #'default'

# ===Basic Param===
data_shuffle = int(False)
model_type = 'vanilla'
gen_init = 'truncated_normal'
dis_init = 'uniform'
samples_num = 10000
batch_size = 64
max_seq_len = 20
gen_lr = 0.01
gen_adv_lr = 1e-4
dis_lr = 1e-4
pre_log_step = 10
adv_log_step = 200

# ===Generator===
ADV_g_step = 1
gen_embed_dim = 32
gen_hidden_dim = 32
mem_slots = 1
num_heads = 2
head_size = 256

# ===Discriminator===
ADV_d_step = 5
dis_embed_dim = 64
dis_hidden_dim = 64
num_rep = 64

# ===Metrics===
use_nll_oracle = int(True)
use_nll_gen = int(False)
use_nll_div = int(True)
use_bleu = int(True)
use_self_bleu = int(True)
use_ppl = int(False)

args = [
    # Program
    '--if_test', if_test,
    '--run_model', run_model,
    '--cuda', CUDA,
    # '--device', gpu_id,   # comment for auto GPU
    '--ora_pretrain', oracle_pretrain,
    '--gen_pretrain', gen_pretrain,
    '--dis_pretrain', dis_pretrain,
    '--mle_epoch', MLE_train_epoch,
    '--adv_epoch', ADV_train_epoch,
    '--tips', tips,

    # Oracle or Real
    '--if_real_data', if_real_data[job_id],
    '--dataset', dataset[job_id],
    '--loss_type', loss_type,
    '--vocab_size', vocab_size[job_id],
    '--temp_adpt', temp_adpt,
    '--temperature', temperature[job_id],

    # Basic Param
    '--shuffle', data_shuffle,
    '--model_type', model_type,
    '--gen_init', gen_init,
    '--dis_init', dis_init,
    '--samples_num', samples_num,
    '--batch_size', batch_size,
    '--max_seq_len', max_seq_len,
    '--gen_lr', gen_lr,
    '--gen_adv_lr', gen_adv_lr,
    '--dis_lr', dis_lr,
    '--pre_log_step', pre_log_step,
    '--adv_log_step', adv_log_step,

    # aug
    '--key' , key,
    '--train_ratio', train_ratio,
    '--trial', trial,
    '--aug', aug,
    '--aug_place', aug_place,

    # Generator
    '--adv_g_step', ADV_g_step,
    '--gen_embed_dim', gen_embed_dim,
    '--gen_hidden_dim', gen_hidden_dim,
    '--mem_slots', mem_slots,
    '--num_heads', num_heads,
    '--head_size', head_size,

    # Discriminator
    '--adv_d_step', ADV_d_step,
    '--dis_embed_dim', dis_embed_dim,
    '--dis_hidden_dim', dis_hidden_dim,
    '--num_rep', num_rep,

    # Metrics
    '--use_nll_oracle', use_nll_oracle,
    '--use_nll_gen', use_nll_gen,
    '--use_nll_div', use_nll_div,
    '--use_bleu', use_bleu,
    '--use_self_bleu', use_self_bleu,
    '--use_ppl', use_ppl,
]

args = list(map(str, args))
my_env = os.environ.copy()
call([executable, scriptname] + args, env=my_env, cwd=rootdir)
