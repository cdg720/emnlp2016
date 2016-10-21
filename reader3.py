from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections, gzip, os

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
