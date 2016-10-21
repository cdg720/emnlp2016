from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils import _build_vocab, nbest_iterator, open_file, unkify

import os

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


def _file_to_word_ids(filename, word2id):
  data = []
  trees = []
  idx2tree = []
  for ts in _generate_nbest(open_file(filename)):
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
  train_path = os.path.join(data_path, "train.gz")
  word_to_id = _build_vocab(train_path)
  nbest_data = _file_to_word_ids(nbest_path, word_to_id)
  return nbest_data, word_to_id
