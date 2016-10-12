from bllipparser import RerankingParser, Tree
from collections import deque
import gzip, math, nltk, sys


def generate_nbest(f):
  nbest = []
  count = 0
  for line in f:
    line = line[:-1]
    if line == '':
      continue
    if count == 0: # the very first
      count = int(line.split()[0])
    elif line.startswith('('):
      nbest.append({'ptb': line})
      count -= 1
      if count == 0:
        yield nbest
        nbest = []


def open_file(path):
  if path.endswith('.gz'):
    return gzip.open(path, 'rb')
  else:
    return open(path, 'r')


def ptb(t, words):
  forms = []
  ptb_recurse(t, words, forms)
  return ' ' + ' '.join(forms) + ' '


def ptb_recurse(t, words, forms):
  forms.append('(' + t.label())
  for child in t: 
    if child.height() == 2:
      if child[0].lower() not in words:
        forms.append(unkify(child[0]))
      else:
        forms.append(child[0].lower())
    else:
      ptb_recurse(child, words, forms)
  forms.append(')' + t.label())


def read_vocab(path):
  words = {}
  for line in open_file(path):
    words[line[:-1]] = len(words)
  return words


def remove_duplicates(nbest):
  new_nbest = []
  for t in nbest:
    good = True
    for new_t in new_nbest:
      if t['seq'] == new_t['seq']:
        good = False
        break
    if good:
      new_nbest.append(t)
  return new_nbest


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


if __name__ == '__main__':
  if len(sys.argv) != 3 and len(sys.argv) != 4:
    print 'usage: python traversal.py vocab.gz gold.gz [nbest.gz]'
    sys.exit(0)

  words = read_vocab(sys.argv[1])
  if len(sys.argv) == 3:
    for line in open_file(sys.argv[2]):
      t = nltk.Tree.fromstring(line[:-1])[0]
      print ptb(t, words)
  else:
    rrp = RerankingParser()
    parser = 'data/WSJ-PTB3/parser'
    rrp.load_parser_model(parser)
    for gold, nbest in zip(open_file(sys.argv[2]),
                           generate_nbest(open_file(sys.argv[3]))):
      for tree in nbest:
        t = nltk.Tree.fromstring(tree['ptb'])[0]
        tree['seq'] = ptb(t, words)
      nbest = remove_duplicates(nbest)
      gold = Tree(gold)
      print len(nbest)
      for t in nbest:
        scores = Tree(t['ptb']).evaluate(gold)
        print scores['gold'], scores['test'], scores['matched']
        print t['seq']
