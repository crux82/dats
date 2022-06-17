"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

  Licensed under the Apache License, Version 2.0 (the "License").
  You may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
import argparse
import numpy as np
import random
import pandas as pd

from sklearn.model_selection import train_test_split
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='Input File Path', required=True)
parser.add_argument('-m', '--model_dir', help='Model output dir', default='output2')
parser.add_argument('-th', '--similarity_threshold', default=0.0, type=float, help='Cosine similarity Threshold')
parser.add_argument('-split', '--dev_split', default=0.1, type=float,
                    help='Percentage of the dataset used to evaluate the early stopping')
parser.add_argument('-epoch', '--epoch', default=1, type=int, help='number of training epochs')
parser.add_argument('-p', '--patience', help='Number of patience steps', default=3, type=int, )
parser.add_argument('-lr', '--learning_rate', default=2e-5, type=float, help='learning rate')
parser.add_argument('-wr', '--warmup_ratio', default=0.1, type=float, help='The warmup ratio for the linear scheduler')
parser.add_argument('-model_name', '--model_name', help='Pretrained Model', default='facebook/bart-base')
parser.add_argument('-model_type', '--model_type', help='Pretrained Model', default='bart')
parser.add_argument('-bs', '--batch_size', help='Batch size', default=32, type=int)
parser.add_argument('-ng', '--num_gpus', help='Number of GPUs to use', default=1, type=int)
parser.add_argument('--seed', default=23, type=int, help='The seed used in the random process generator')

args = parser.parse_args()

input_file_path = args.input
model_output_dir = args.model_dir
model_name = args.model_name
model_type = args.model_type
output_file_path = args.model_dir
patience = args.patience
epoch = args.epoch
learning_rate = args.learning_rate
warmup_ratio = args.warmup_ratio
batch_size = args.batch_size
similarity_threshold = args.similarity_threshold
num_gpus = args.num_gpus
dev_split = args.dev_split

##Set random values
seed_val = args.seed
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_val)

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)

dataset_df = pd.read_csv(input_file_path, header=None, usecols=[0, 1, 2, 3, 4, 5, 6, 7],
                         names=['0', '1', 'input_class', 'input_text', 'bert_sim', '5', 'target_text', '7'], sep="\t").astype(str)

# dataset_df = dataset_df.rename(
#     columns={"2": "input_class", "3": "input_text", "4": "bert_sim", "6": "target_text"}
# )

dataset_df['input_class'] = dataset_df['input_class'].str.replace(":", "_")

dataset_df['input_text'] = dataset_df['input_class'].astype(str) + " " + dataset_df['input_text']
dataset_df['target_text'] = dataset_df['input_class'].astype(str) + " " + dataset_df['target_text']

dataset_df["bert_sim"] = dataset_df["bert_sim"].astype(float)

df_mask = dataset_df["bert_sim"] > similarity_threshold
dataset_df = dataset_df[df_mask]

dataset_df = dataset_df[["input_text", "target_text"]]

# split according to the input_text column
input_text_df_uniq = dataset_df["input_text"].drop_duplicates()
input_text_df_uniq_train, input_text_df_uniq_eval = train_test_split(input_text_df_uniq, test_size=dev_split)
train_df = dataset_df[np.isin(dataset_df["input_text"], input_text_df_uniq_train)]
eval_df = dataset_df[np.isin(dataset_df["input_text"], input_text_df_uniq_eval)]

model_args = Seq2SeqArgs()
model_args.eval_batch_size = 16  # 64
model_args.evaluate_during_training = True  # True
model_args.evaluate_during_training_steps = 2500
model_args.evaluate_during_training_verbose = True
model_args.fp16 = False
model_args.learning_rate = learning_rate
model_args.max_length = 64  # 128
model_args.max_seq_length = 64  # 128
model_args.num_train_epochs = epoch
model_args.optimizer = "AdamW"
model_args.scheduler = "linear_schedule_with_warmup"
model_args.warmup_ratio = warmup_ratio
model_args.use_early_stopping = True
model_args.early_stopping_consider_epochs = True
model_args.early_stopping_metric = "eval_loss"
model_args.early_stopping_metric_minimize = True
model_args.early_stopping_delta = 0.01
model_args.early_stopping_patience = patience
model_args.overwrite_output_dir = True
model_args.reprocess_input_data = True
model_args.save_steps = -1  # check
model_args.save_model_every_epoch = False  # check
model_args.save_eval_checkpoints = False
model_args.save_steps = -1
model_args.train_batch_size = batch_size
model_args.use_multiprocessing = False
model_args.manual_seed = seed_val
model_args.output_dir = model_output_dir
model_args.best_model_dir = model_output_dir + "/best_model"
model_args.cache_dir = model_output_dir + "/cache_dir"
model_args.n_gpu = num_gpus

model = Seq2SeqModel(
    encoder_decoder_type=model_type,
    encoder_decoder_name=model_name,
    args=model_args,
)

model.train_model(train_df, eval_data=eval_df, output_dir=model_output_dir)
