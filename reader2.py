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


def _generate_nbest(f):
  nbest = []
  count = 0
  for line in f:
    line = line[:-1]
    if line == '':
      continue
    if count == 0:
      count = int(line.split()[0])
    elif line.startswith('('):
      nbest.append({'ptb': line})
      count -= 1
      if count == 0:
        yield nbest
        nbest = []


def _remove_duplicates(nbest):
  new_nbest = []
  seqs = set()
  for t in nbest:
    if t['seq'] not in seqs:
      seqs.add(t['seq'])
      new_nbest.append(t)
  return new_nbest


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


def _file_to_word_ids(filename, word2id):
  data = []
  trees = []
  idx2tree = []
  for ts in _generate_nbest(gzip.open(filename, 'rb')):
    for t in ts:
      t['seq'] = process_tree(t['ptb'], word2id)
    ts = _remove_duplicates(ts)
    nbest = []
    for t in ts:
      nums = [word2id[word] for word in t['seq'].split() + ['<eos>']]
      for i in xrange(len(nums)):
        idx2tree.append((len(trees), len(nbest)))
      nbest.append(t['ptb'])
      data.extend(nums)
    trees.append(nbest)
  return {'data': data, 'trees': trees, 'idx2tree': idx2tree}


def unkify(ws):
  uk = 'unk'
  sz = len(ws)-1
  if ws[0].isupper():
    uk = 'c' + uk
  if ws[0].isdigit() and ws[sz].isdigit():
    uk = uk + 'n'
  elif sz <= 2:
    pass
  elif ws[sz-2:sz+1] == 'ing':
    uk = uk + 'ing'
  elif ws[sz-1:sz+1] == 'ed':
    uk = uk + 'ed'
  elif ws[sz-1:sz+1] == 'ly':
    uk = uk + 'ly'
  elif ws[sz] == 's':
    uk = uk + 's'
  elif ws[sz-2:sz+1] == 'est':
    uk = uk + 'est'
  elif ws[sz-1:sz+1] == 'er':
    uk = uk + 'ER'
  elif ws[sz-2:sz+1] == 'ion':
    uk = uk + 'ion'
  elif ws[sz-2:sz+1] == 'ory':
    uk = uk + 'ory'
  elif ws[0:2] == 'un':
    uk = 'un' + uk
  elif ws[sz-1:sz+1] == 'al':
    uk = uk + 'al'
  else:
    for i in xrange(sz):
      if ws[i] == '-':
        uk = uk + '-'
        break
      elif ws[i] == '.':
        uk = uk + '.'
        break
  return '<' + uk + '>'


def process_tree(line, words, tags=False):
  tokens = line.replace(')', ' )').split()
  nonterminals = []
  new_tokens = []
  pop = False
  ind = 0
  for token in tokens:
    if token.startswith('('): # open paren
      new_token = token[1:]
      nonterminals.append(new_token)
      new_tokens.append(token)
    elif token == ')': # close paren
      if pop: # preterminal
        pop = False
      else: # nonterminal
        new_token = ')' + nonterminals.pop()
        new_tokens.append(new_token)
    else: # word
      if not tags:
        tag = '(' + nonterminals.pop() # pop preterminal
        new_tokens.pop()
        pop = True
      if token.lower() in words:
        new_tokens.append(token.lower())
      else:
        new_tokens.append(unkify(token))
  return ' ' + ' '.join(new_tokens[1:-1]) + ' '


def ptb_raw_data(data_path=None, nbest_path=None):
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
  word_to_id = _build_vocab(train_path)
  nbest_data = _file_to_word_ids(nbest_path, word_to_id)
  return nbest_data, word_to_id


def ptb_iterator(raw_data, batch_size, num_steps, idx2tree, eos):
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
