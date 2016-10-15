from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections, gzip, os

import numpy as np
import tensorflow as tf


def _build_vocab(filename):
  word2id = {}
  for line in gzip.open(filename, 'rb'):
    word, i = line.split()
    word2id[word] = int(i)
  return word2id


# read train and dev
def _file_to_word_ids(filename):
  return [int(x) for x in gzip.open(filename, 'rb').read().split()]


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
        gold = int(tmp[0])
        test = int(tmp[1])
        matched = int(tmp[2])        
      else:            
        line = line.replace('\n', '<eos>').split()
        line = [word_to_id[word] for word in line]
        for i in xrange(len(line)):
          idx2tree.append((len(scores), len(nbest)))
        nbest.append({'gold': gold, 'test': test, 'matched': matched})
        count -= 1
        data.extend(line)
        if count == 0:
          scores.append(nbest)
          nbest = []
  return {'data': data, 'scores': scores, 'idx2tree': idx2tree}


# read silver data
def _file_to_word_ids3(filename):
  for line in gzip.open(filename, 'rb'):
    yield [int(x) for x in line.split()] 


def ptb_raw_data(data_path=None):
  train_path = os.path.join(data_path, "train.gz")
  silver_path = os.path.join(data_path, 'silver2.gz')
  valid_path = os.path.join(data_path, "dev.gz")
  vocab_path = os.path.join(data_path, 'vocab.gz')
  valid_nbest_path = os.path.join(data_path, "dev_nbest.gz")

  word_to_id = _build_vocab(vocab_path)
  train_data = _file_to_word_ids(train_path)
  valid_data = _file_to_word_ids(valid_path)
  valid_nbest_data = _file_to_word_ids2(valid_nbest_path, word_to_id)
  return train_data, silver_path, valid_data, valid_nbest_data, word_to_id


def ptb_iterator(raw_data, batch_size, num_steps):
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
