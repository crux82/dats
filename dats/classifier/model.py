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

import torch.nn as nn
from transformers import *


class Classifier(nn.Module):
    def __init__(self, model_name, num_labels=2, dropout_rate=0.1):
        super(Classifier, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        self.cls_size = int(config.hidden_size)
        self.input_dropout = nn.Dropout(p=dropout_rate)
        self.fully_connected_layer = nn.Linear(self.cls_size, num_labels)

    def forward(self, input_ids, attention_mask):
        model_outputs = self.encoder(input_ids, attention_mask)
        encoded_cls = model_outputs.last_hidden_state[:, 0]
        encoded_cls_dropout = self.input_dropout(encoded_cls)
        logits = self.fully_connected_layer(encoded_cls_dropout)
        return logits, encoded_cls
