# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gzip
import os

import numpy as np
import tensorflow as tf


def _read_words(filename):
  with gzip.open(filename, 'rb') as f:
    return f.read().replace('\n', '<eos>').split()
  

def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data]


# read nbest
def _file_to_word_ids2(filename, word_to_id):
  data = []
  scores = []
  nbest = []
  idx2tree = []
  count = 0
  with gzip.open(filename, 'rb') as f:
    for line in f:
      if count == 0:
        count = int(line)
      elif line.startswith('-'):
        tmp = line.split()
        parser_score = float(tmp[0])
        gold = int(tmp[1])
        test = int(tmp[2])
        matched = int(tmp[3])        
      else:            
        line = line.replace('\n', '<eos>').split()
        line = [word_to_id[word] for word in line]
        for i in xrange(len(line)):
          idx2tree.append((len(scores), len(nbest)))
        nbest.append({'parser_score': parser_score, 'gold': gold,
                      'test': test, 'matched': matched})
        count -= 1
        data.extend(line)
        if count == 0:
          scores.append(nbest)
          nbest = []
  return {'data': data, 'scores': scores, 'idx2tree': idx2tree}


def ptb_raw_data(data_path=None):
  """Load PTB raw data from data directory "data_path".

  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.

  The PTB dataset comes from Tomas Mikolov's webpage:

  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, "train.gz")
  valid_path = os.path.join(data_path, "dev.gz")
  test_path = os.path.join(data_path, "dev.gz")
  valid_nbest_path = os.path.join(data_path, "dev_nbest.gz")
  test_nbest_path = os.path.join(data_path, "dev_nbest.gz")

  word_to_id = _build_vocab(train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  valid_nbest_data = _file_to_word_ids2(valid_nbest_path, word_to_id)
  test_nbest_data = _file_to_word_ids2(test_nbest_path, word_to_id)
  return train_data, valid_data, test_data, \
    valid_nbest_data, test_nbest_data, word_to_id


def ptb_raw_data2(data_path=None):
  train_path = os.path.join(data_path, "train.gz")
  valid_path = os.path.join(data_path, "dev.gz")
  test_path = os.path.join(data_path, "dev.gz")
  valid_nbest_path = os.path.join(data_path, "dev_nbest.gz")
  test_nbest_path = os.path.join(data_path, "dev_nbest.gz")

  word_to_id = _build_vocab(train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)
  return train_data, valid_data, test_data, \
    valid_nbest_path, test_nbest_path, word_to_id, vocabulary


def ptb_iterator(raw_data, batch_size, num_steps):
  """Iterate on the raw PTB data.

  This generates batch_size pointers into the raw PTB data, and allows
  minibatch iteration along these pointers.

  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.

  Yields:
    Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
    The second element of the tuple is the same data time-shifted to the
    right by one.

  Raises:
    ValueError: if batch_size or num_steps are too high.
  """
  raw_data = np.array(raw_data, dtype=np.int32)

  data_len = len(raw_data)
  batch_len = data_len // batch_size
  data = np.zeros([batch_size, batch_len], dtype=np.int32)
  for i in range(batch_size):
    data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

  epoch_size = (batch_len - 1) // num_steps

  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

  for i in range(epoch_size):
    x = data[:, i*num_steps:(i+1)*num_steps]
    y = data[:, i*num_steps+1:(i+1)*num_steps+1]
    yield (x, y)
    

# iterator used for nbest
def ptb_iterator2(raw_data, batch_size, num_steps, idx2tree, eos):
  dummy1 = 0
  dummy2 = (-1, -1)
  remainder = len(raw_data) % batch_size
  if remainder != 0:
    raw_data = raw_data + [dummy1 for x in xrange(batch_size - remainder)]
    idx2tree = idx2tree + [dummy2 for x in xrange(batch_size - remainder)]
  raw_data = np.array(raw_data, dtype=np.int32)

  data_len = len(raw_data)
  batch_len = data_len // batch_size
  remainder = (data_len // batch_size) % num_steps
    
  data = np.zeros([batch_size, batch_len + num_steps - remainder + 1],
                  dtype=np.int32)
  for i in range(batch_size):
    data[i, 1:batch_len+1] = raw_data[batch_len * i:batch_len * (i + 1)]
    if i == 0:
      data[i, 0] = eos
    else:
      data[i, 0] = raw_data[batch_len - 1]        
  idx2tree = np.array(idx2tree, dtype=np.dtype('int, int'))
  tree = np.zeros([batch_size, batch_len + num_steps - remainder],
                  dtype=np.dtype('int, int'))
  for i in range(batch_size):
    tree[i, :batch_len] = idx2tree[batch_len * i:batch_len * (i + 1)]
    tree[i, batch_len:] = [dummy2 for x in xrange(num_steps - remainder)]

  epoch_size = (batch_len + num_steps - remainder) // num_steps

  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

  for i in range(epoch_size):
    x = data[:, i*num_steps:(i+1)*num_steps]
    y = data[:, i*num_steps+1:(i+1)*num_steps+1]
    z = tree[:, i*num_steps:(i+1)*num_steps]
    yield (x, y, z)
