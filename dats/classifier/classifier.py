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

import random
import time

import argparse

from model import *

from dats.utils.data import *


# Check the GPU
if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# --------------------------------------------
#  Input Parameters from the commandline
# --------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-tr','--train', nargs='+', help='Train File Path', required=True)
parser.add_argument('-te','--test', help='Test File Path', required=True)
parser.add_argument('-dev','--dev', help='Dev File Path' ,required=True)
parser.add_argument('-m','--output_model', help='Name of the output model', required=True)
parser.add_argument('-bs', '--batch_size', default=64, type=int, help='batch size')
parser.add_argument('-max_len', '--max_seq_length', default=64, type=int, help='The maximum length to be considered in input')
parser.add_argument('-max_gen', '--max_generated_examples', default=10, type=int, help='The maximum number of generated similar examples')
parser.add_argument('-nfold', '--nfold', default=2, type=int, help='The maximum number of fold used to split the generated similar examples')
parser.add_argument('-lr', '--learning_rate', default=2e-5, type=float, help='learning rate')
parser.add_argument('-p','--patience', help='Number of patience steps', default=3, type=int,)
parser.add_argument('-epochs', '--epochs', default=10, type=int, help='number of training epochs')
parser.add_argument('-model','--model_name', help='Pretrained Model', default="bert-base-uncased")
parser.add_argument('-tr_sim','--train_exp_file_path', help='Path of the file containing the train explanatory model', default="train_expl.txt")
parser.add_argument('-te_sim','--test_exp_file_path', help='Path of the file containing the test explanatory model', default="test_expl.txt")
parser.add_argument('--seed', default=23, type=int, help='The seed used in the random process generator')


args = parser.parse_args()

print("Input Parameters:")
print(args)

# --------------------------------------------
#  Input Parameters
# --------------------------------------------
train_filenames = args.train
dev_filename = args.dev
test_filename = args.test
output_model_name = args.output_model
batch_size = args.batch_size
max_seq_length = args.max_seq_length
learning_rate = args.learning_rate
train_exp_file_path = args.train_exp_file_path
test_exp_file_path = args.test_exp_file_path
max_generated_examples = args.max_generated_examples
model_name = args.model_name
nfold = args.nfold
patience = args.patience
epochs = args.epochs

## Set random values
seed_val = args.seed
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(seed_val)

# BERT output dropout
out_dropout_rate = 0.1
# Print a log each n steps
print_each_n_step = 10

# --------------------------------
# Load the training material
# --------------------------------
train_examples_list = []
train_labels_list = []

for train_filename in train_filenames:
    train_examples, train_labels = load_examples(train_filename)

    train_examples_list.append(train_examples)
    train_labels_list.append(train_labels)

dev_examples, dev_labels = load_examples(dev_filename)
_, test_labels = load_examples(test_filename)

# Select the target classes
label_list = set()
for train_labels in train_labels_list:
    label_list = label_list.union(train_labels)

label_list = list(label_list.union(dev_labels).union(test_labels))
label_list.sort()
# Let us print the labels used in the dataset
print("Target Labels:\t" + str(label_list))
print("Number of Labels:\t" + str(len(label_list)))

# Initialize a map to associate labels to the dimension of the embedding
# produced by the classifier
label_to_id_map = {}
id_to_label_map = {}
for (i, label) in enumerate(label_list):
  label_to_id_map[label] = i
  id_to_label_map[i] = label

# Shuffle and split the training material in train/dev

# Define a Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_dataloader_list = []
# Build the Train Dataloader
for train_examples in train_examples_list:
    train_dataloader = generate_data_loader(train_examples, label_to_id_map, tokenizer, batch_size, do_shuffle=True, max_seq_length=max_seq_length)
    train_dataloader_list.append(train_dataloader)
# Build the Development Dataloader
dev_dataloader = generate_data_loader(dev_examples, label_to_id_map, tokenizer, batch_size, do_shuffle=True, max_seq_length=max_seq_length)

for train_examples in train_examples_list:
    print("Number of training examples:\t"+ str(len(train_examples)))

print("Number of development examples:\t"+ str(len(dev_examples)))

classifier = Classifier(model_name, num_labels=len(label_list), dropout_rate=out_dropout_rate)

# Put everything in the GPU if available
if torch.cuda.is_available():
    classifier.cuda()

optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate)
training_stats = []

nll_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

total_t0 = time.time()

dataset_idx = 0
best_dev_accuracy = -1

for train_dataloader in train_dataloader_list:
    actual_patience = patience

    if len(train_examples_list) == 1 or dataset_idx == 1:
        real_epochs = epochs
    else:
        real_epochs = 1
    dataset_idx = dataset_idx + 1

    for epoch_i in range(0, real_epochs):
        # ========================================
        #               Training Loop
        # ========================================
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()

        # Reset the loss
        train_loss = 0

        classifier.train()

        for step, batch in enumerate(train_dataloader):
            if step % print_each_n_step == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader.
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # clear the gradients
            optimizer.zero_grad()

            train_logits, _ = classifier(b_input_ids, b_input_mask)
            loss = nll_loss(train_logits, b_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)
        training_time = format_time(time.time() - t0)
        print("")
        print("  Average training loss: {0:.3f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

        # ========================================
        #     Evaluate on Development set
        # ========================================
        print("")
        print("Running Evaluation on Development Set")

        t0 = time.time()
        classifier.eval()

        avg_dev_loss, dev_accuracy = evaluate(dev_dataloader, tokenizer, classifier)
        test_time = format_time(time.time() - t0)

        print("  Accuracy: {0:.3f}".format(dev_accuracy))
        print("  Test Loss: {0:.3f}".format(avg_dev_loss))
        print("  Test took: {:}".format(test_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_dev_loss,
                'Valid. Accur.': dev_accuracy,
                'Training Time': training_time,
                'Test Time': test_time
            }
        )

        if dev_accuracy > best_dev_accuracy:
          best_dev_accuracy = dev_accuracy
          actual_patience = patience
          torch.save(classifier, output_model_name)
          print("\n  Saving the model during epoch " + str(epoch_i + 1))
          print("\n  Resetting patience to " + str(actual_patience))
          print("  Actual Best Validation Accuracy: {0:.3f}".format(best_dev_accuracy))
        else:
          actual_patience = actual_patience - 1
          print("\n  Setting patience to " + str(actual_patience))

        if actual_patience == 0:
            print("\n  Min patience reached at epoch " + str(epoch_i + 1))
            break

"""Print some statistics about the training"""

train_losses = []
val_losses = []
train_acc = []
val_acc = []

for stat in training_stats:
  train_losses.append(stat["Training Loss"])
  val_losses.append(stat["Valid. Loss"])
  val_acc.append(stat["Valid. Accur."])
  print(stat)

print("\nTraining complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

# Reload the best model
classifier = torch.load(output_model_name)

# Reload train examples
train_examples, train_labels = load_examples(train_filename)

if max_generated_examples > 0:
    if nfold == 1:
        produce_similar_examples(train_exp_file_path, train_examples, train_examples, batch_size, max_seq_length, label_to_id_map, label_list, tokenizer, classifier, max_examples=max_generated_examples)
    else:
        train_example_fold_list = split(train_examples, nfold)
        for fold_idx, train_examples_in_fold in enumerate(train_example_fold_list):
            produce_similar_examples(train_exp_file_path+"_"+str(fold_idx+1), train_examples_in_fold, train_examples, batch_size, max_seq_length, label_to_id_map, label_list, tokenizer, classifier, max_examples=max_generated_examples)

# Load the test data, generate the data loade and evaluate
test_examples, _ = load_examples(test_filename)
test_dataloader = generate_data_loader(test_examples, label_to_id_map, tokenizer, batch_size, do_shuffle=False, max_seq_length=max_seq_length)
avg_test_loss, test_accuracy = evaluate(test_dataloader, tokenizer, classifier, print_classification_output=True, print_result_summary=False, id_to_label_map=id_to_label_map, label_list=label_list)

if max_generated_examples > 0:
    produce_similar_examples(test_exp_file_path, test_examples, train_examples, batch_size, max_seq_length, label_to_id_map, label_list, tokenizer, classifier, max_examples=max_generated_examples)

print("\n\nFINAL\tAccuracy\t"+ train_filename + "\tVS\t" + test_filename +"\tdev\t{0:.3f}".format(best_dev_accuracy)+"\ttest\t{0:.3f}".format(test_accuracy))
print("  Test Loss: {0:.3f}".format(avg_test_loss))
