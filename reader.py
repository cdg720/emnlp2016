from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils import _build_vocab, _read_words, open_file

import gzip, os

import numpy as np
import tensorflow as tf


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
  with open_file(filename) as f:
    for line in f:
      if count == 0:
        count = int(line)
      elif not line.startswith(' '):
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


def ptb_raw_data(data_path=None):
  train_path = os.path.join(data_path, "train.gz")
  valid_path = os.path.join(data_path, "dev.gz")
  valid_nbest_path = os.path.join(data_path, "dev_nbest.gz")

  word_to_id = _build_vocab(train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  valid_nbest_data = _file_to_word_ids2(valid_nbest_path, word_to_id)
  return train_data, valid_data, valid_nbest_data, word_to_id
