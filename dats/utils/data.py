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

import torch
import torch.nn as nn

import datetime
import sys
import numpy as np
import re

from transformers import *

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def load_examples(input_file):
  """
    Loading examples in txt file where we assume the first token (space based) is the category and the rest is the text instance.
  """
  examples = []
  labels = []

  with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
      contents = f.read()
      file_as_list = contents.splitlines()
      for line in file_as_list:
          split = line.split(" ")
          question = ' '.join(split[1:])
          labels.append(split[0])
          examples.append((question, split[0]))

  return examples, set(labels)


def generate_data_loader(examples, label_map, tokenizer, batch_size, do_shuffle = False, max_seq_length=64):
  """
    Generate the data loader for training
  """
  input_ids = []
  input_mask_array = []
  label_id_array = []

  # Tokenization
  for (text, label) in examples:
    # Using huggingface tokenizer
    encoded_sent = tokenizer.encode_plus(text, add_special_tokens=True, max_length=max_seq_length, padding='max_length', truncation=True)
    input_ids.append(encoded_sent['input_ids'])
    input_mask_array.append(encoded_sent['attention_mask'])

    # converts labels to IDs
    id = -1
    if label in label_map:
      id = label_map[label]
    label_id_array.append(id)

  # Convert to Tensor which are used in PyTorch
  input_ids = torch.tensor(input_ids)
  input_mask_array = torch.tensor(input_mask_array)
  label_id_array = torch.tensor(label_id_array, dtype=torch.long)

  # Building the TensorDataset
  dataset = TensorDataset(input_ids, input_mask_array, label_id_array)

  if do_shuffle:
    # this will shuffle examples each time a new batch is required
    sampler = RandomSampler
  else:
    sampler = SequentialSampler

  return DataLoader(dataset, sampler = sampler(dataset), batch_size = batch_size)


def evaluate(dataloader, tokenizer, classifier, print_classification_output=False, print_result_summary=False, id_to_label_map=None, label_list = None):
  """
    Evaluation function.
  """
  total_loss = 0
  gold_classes = []
  system_classes = []

  nll_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

  # Checking whether cuda is available
  if torch.cuda.is_available():
      device = torch.device("cuda")
  else:
      device = torch.device("cpu")

  if print_classification_output:
      print("\n------------------------")
      print("  Classification outcomes")
      print("is_correct\tgold_label\tsystem_label\ttext")
      print("------------------------")

  # For each batch of examples from the input dataloader
  for batch in dataloader:
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_labels = batch[2].to(device)

    with torch.no_grad():
      # Classify the batch
      logits, _ = classifier(b_input_ids, b_input_mask)
      # Evaluate the loss.
      total_loss += nll_loss(logits, b_labels)

    # Accumulate the predictions and the input labels
    _, preds = torch.max(logits, 1)
    system_classes += preds.detach().cpu()
    gold_classes += b_labels.detach().cpu()

    # Print the output of the classification for each input element
    if print_classification_output:
      for ex_id in range(len(b_input_mask)):
        input_strings = tokenizer.decode(b_input_ids[ex_id], skip_special_tokens=True)
        # convert class id to the real label
        predicted_label = id_to_label_map[preds[ex_id].item()]
        gold_standard_label = "UNKNOWN"
        # convert the gold standard class ID into a real label
        if b_labels[ex_id].item() in id_to_label_map:
          gold_standard_label = id_to_label_map[b_labels[ex_id].item()]
        # put the prefix "[OK]" if the classification is correct
        output = '[OK]' if predicted_label == gold_standard_label else '[NO]'
        # print the output
        print(output+"\t"+gold_standard_label+"\t"+predicted_label+"\t"+input_strings)

  avg_loss = total_loss / len(dataloader)
  avg_loss = avg_loss.item()

  system_classes = torch.stack(system_classes).numpy()
  gold_classes = torch.stack(gold_classes).numpy()
  accuracy = np.sum(system_classes == gold_classes) / len(system_classes)

  if print_result_summary:
    print("\n------------------------")
    print("  Summary")
    print("------------------------")
    #remove unused classes in the test material
    filtered_label_list = []
    for i in range(len(label_list)):
      if i in gold_classes:
        filtered_label_list.append(id_to_label_map[i])
    print(classification_report(gold_classes, system_classes, digits=3, target_names=filtered_label_list))

    print("\n------------------------")
    print("  Confusion Matrix")
    print("------------------------")
    conf_mat = confusion_matrix(gold_classes, system_classes)
    for row_id in range(len(conf_mat)):
      print(filtered_label_list[row_id]+"\t"+str(conf_mat[row_id]))

  return avg_loss, accuracy


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def split_in_batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def produce_similar_examples(output_file_path, target_examples, train_examples, batch_size, max_seq_length, label_to_id_map, label_list, tokenizer, classifier, max_examples=5):
  train_dataloader = generate_data_loader(train_examples, label_to_id_map, tokenizer, batch_size, do_shuffle = False, max_seq_length = max_seq_length)
  target_dataloader = generate_data_loader(target_examples, label_to_id_map, tokenizer, batch_size, do_shuffle = False, max_seq_length = max_seq_length)

  train_cls_list = []

  softmax = nn.Softmax(dim=1)

  if torch.cuda.is_available():
      device = torch.device("cuda")
  else:
      device = torch.device("cpu")

  batch_counter = 1
  for batch in train_dataloader:
    print("Processing batch " + str(batch_counter))
    batch_counter = batch_counter + 1
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_labels = batch[2].to(device)
    with torch.no_grad():
      _, cls = classifier(b_input_ids, b_input_mask)
      train_cls_list.extend(cls)

  train_cls_tensor = torch.stack(train_cls_list)

  batch_counter = 1
  with open(output_file_path, 'w', encoding='utf8') as f:
    # Divide in batch target examples and process to reduce memory requirements
    for batch_target_examples in split_in_batch(target_examples, batch_size):
      print("Processing batch " + str(batch_counter))
      batch_counter = batch_counter + 1

      batch_target_dataloader = generate_data_loader(batch_target_examples, label_to_id_map, tokenizer, batch_size, do_shuffle = False, max_seq_length = max_seq_length)

      target_cls_list = []
      target_logits = []

      for batch in batch_target_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        with torch.no_grad():
          logits, cls = classifier(b_input_ids, b_input_mask)
          target_cls_list.extend(cls)
          target_logits.extend(softmax(logits))

      target_cls_tensor = torch.stack(target_cls_list)

      cosim_matrix = sim_matrix(target_cls_tensor, train_cls_tensor)
      cosim_indices = torch.argsort( - cosim_matrix, dim=1)

      for i, example in enumerate(batch_target_examples):
        correct_label = example[1]

        ordere_predicted_label_id = torch.argsort( - target_logits[i], dim=0).cpu().detach().numpy()

        first_predicted_label = label_list[ordere_predicted_label_id[0]]

        first_incorrect_label = first_predicted_label
        if first_incorrect_label == correct_label:
          first_incorrect_label = label_list[ordere_predicted_label_id[1]]

        for label in [correct_label, first_incorrect_label]:
          ex_for_label_list = []
          cosim_for_example = []
          for j in range(len(train_examples)):
            cosim_index = cosim_indices[i][j]

            if cosim_matrix[i][cosim_index] > 0.999:
              continue

            candidate_ex = train_examples[cosim_index]
            candidate_cat = candidate_ex[1]
            if candidate_cat == label:
              ex_for_label_list.append(candidate_ex)
              cosim_for_example.append(cosim_matrix[i][cosim_index])

            if len(ex_for_label_list) >= max_examples:
              break

          counter = 1
          for idxe, ex_for_label in enumerate(ex_for_label_list):
            cosim = cosim_for_example[idxe]
            cand_label = ex_for_label[1]

            if cand_label == correct_label:
              f.write('{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}\n'.format(i,counter,correct_label,example[0],cosim,cand_label,ex_for_label[0],first_predicted_label))
              counter = counter + 1

def clear(str):
    res = str
    res = res.replace("?", " ?")
    res = res.replace("  ", " ")
    res = res.replace("'s", " 's")
    res = re.sub(" +", " ", res)
    return res