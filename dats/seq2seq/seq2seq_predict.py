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

from datetime import datetime
import argparse
import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split

from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs

from dats.utils.data import *

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def clean_qc(input_str):
    res = input_str
    input_str = input_str.replace("?"," ?")
    input_str = input_str.replace("''"," ?")


parser = argparse.ArgumentParser()
parser.add_argument('-i','--input', help='Input File Path',required=True)
parser.add_argument('-m','--model_dir', help='Bart Model Fila path',default='outputs')
parser.add_argument('-model_type','--model_type', help='Pretrained Model',default='bart')
parser.add_argument('-o','--output_data', help='Output augmented data file path',default='augmented_data.txt')
parser.add_argument('-n','--num_generated', help='Number of generated examples', default=1, type=int)

# Generation params
parser.add_argument('-nb','--num_beams', help='Size of the beam (Default: 1)', default=1, type=int)
parser.add_argument('-ds','--do_sample', help='Whether using or not sampling (Default: False)', default=False, type=bool)
parser.add_argument('-tk','--top_k', help='k parameter for the Top-K Sampling', default=-1, type=int)
parser.add_argument('-tp','--top_p', help='p parameter for the Top-p (Nucleous) Sampling', default=-1, type=float)

parser.add_argument('--seed', default=23, type=int, help='The seed used in the random process generator')


args = parser.parse_args()

input_file_path = args.input
model_type = args.model_type
model_dir = args.model_dir
output_file_path = args.output_data
num_generated = args.num_generated

# Generation params
num_generated = args.num_generated
do_sample = args.do_sample
num_beams = args.num_beams
top_k = args.top_k
top_p = args.top_p

##Set random values
seed_val = args.seed
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(seed_val)

examples, labels = load_examples(input_file_path)

input_strings = []
input_classes = []

for example in examples:
    input_strings.append(example[0])
    input_classes.append(example[1])

data = {'input_text': input_strings, 'input_class': input_classes, 'target_text': input_strings}

df = pd.DataFrame(data)

model_args = Seq2SeqArgs()
model_args.top_k = top_k
model_args.top_p = top_p
model_args.do_sample = do_sample
model_args.forced_bos_token_id = None
model_args.use_multiprocessing = True
model_args.num_return_sequences = num_generated
model_args.num_beams = num_beams
model_args.eval_batch_size = 16
model_args.max_length = 64
model_args.max_seq_length = 64

model = Seq2SeqModel(
    encoder_decoder_type=model_type,
    encoder_decoder_name=model_dir,
    args=model_args
)

df["tmp_col"] = df['input_class'].astype(str) + " " + df['input_text']
classes = df["input_class"].tolist()
input_texts = df["tmp_col"].tolist()

to_predict = [
    str(input_text)
    for input_text in df["tmp_col"].tolist()
]

preds = model.predict(to_predict)

generated_sentences = set()

# select from the predictions all sentences. If the same sentence is assigned
with open(output_file_path, "w") as f:
    for i, input_text in enumerate(input_texts):
        pred = preds[i]

        if isinstance(pred, str):
            pred = [pred]
        # here the list of predicted examples. The size is equal to "num_generated"
        pred = list(set(pred))

        for pred_elem in pred:
            split = pred_elem.split(" ")

            # check if the generated example is associated to another class and discard
            if split[0] != classes[i]:
                continue

            generated_example = ' '.join(split[1:])
            generated_example = clear(generated_example)

            if generated_example not in generated_sentences:
                f.write(classes[i] + "\t" + input_text + "\t" + generated_example + "\n")
                generated_sentences.add(generated_example)
