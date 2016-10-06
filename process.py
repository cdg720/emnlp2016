from bllipparser import RerankingParser, Tree
from collections import defaultdict
from transform import post_order, pre_order
from unk import unkify
import gzip, math, nltk, operator, sys

def generate_nbest(f):
  nbest = []
  count = 0
  for line in f:
    line = line[:-1]
    if line == '':
      continue
    if count == 0: # the very first
      count = int(line.split()[0])
    else:
      if nbest and len(nbest[-1]) == 1:
        nbest[-1]['ptb'] = line
        count -= 1
        if count == 0:
          yield nbest
          nbest = []
      else:
        nbest.append({'parser_score': float(line)})
        

def open_file(path):
  if path.endswith('.gz'):
    return gzip.open(path, 'rb')
  else:
    return open(path, 'r')


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
        new_tokens.append('<unk>')
        # new_tokens.append(unkify.unkify(token))
      
      # berkely
      # if token in words:
      #   new_tokens.append(token)
      # else:
      #   new_tokens.append(unkify.unkify_berkeley(token, ind, token.lower() in words))
      # ind += 1
  return ' ' + ' '.join(new_tokens[1:-1]) + ' '


def read_vocabs(path):
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
        new_t['parser_score'] = new_t['parser_score'] + math.log(1 + 2 ** (t['parser_score'] - new_t['parser_score']), 2)
        good = False
        break
    if good:
      new_nbest.append(t)
  return new_nbest


if __name__ == '__main__':
  if len(sys.argv) == 1:
    print 'usage: python process.py vocabs.gz with_tags order gold.gz [nbest.gz]'
    sys.exit(0)

  with_tags = True if sys.argv[2] == 'True' else False
  words = read_vocabs(sys.argv[1])
  order = sys.argv[3]
  if len(sys.argv) == 5:
    for line in open_file(sys.argv[4]):
      if order == 'tree':
        print process_tree(line[:-1], words, with_tags)
      # elif order == 'pre':
      #   tokens = []
      #   pre_order(nltk.Tree.fromstring(line), tokens, words, with_tags)
      #   print ' ' + ' '.join(tokens[1:]) + ' '
      # elif order == 'post':
      #   tokens = []
      #   post_order(nltk.Tree.fromstring(line), tokens, words, with_tags)
      #   print ' ' + ' '.join(tokens[:-1]) + ' '
      # elif order == 'post':
      #   print ' ' + post_order(nltk.Tree.fromstring(line)[0], words) + ' '
      else:
        print 'shit'
  else:
    rrp = RerankingParser()
    rrp.load_parser_model('/home/dc65/.local/share/bllipparser/OntoNotes-WSJ/parser')
    for gold, nbest in zip(open_file(sys.argv[4]), generate_nbest(open_file(sys.argv[5]))):
      for t in nbest:
        if order == 'tree':
          t['seq'] = process_tree(t['ptb'], words, with_tags)
        # elif order == 'pre':
        #   tokens = []
        #   pre_order(nltk.Tree.fromstring(t['ptb']), tokens, words, with_tags)
        #   t['seq'] =  ' ' + ' '.join(tokens[1:]) + ' '
        # elif order == 'post':
        #   tokens = []
        #   post_order(nltk.Tree.fromstring(t['ptb']), tokens, words, with_tags)
        #   t['seq'] = ' ' + ' '.join(tokens[:-1]) + ' '
        # elif order == 'post':
        #   t['seq'] = 'i'
        else:
          print 'shit'
      nbest = remove_duplicates(nbest)
      gold = Tree(gold)
      print len(nbest)
      for t in nbest:
        scores = Tree(t['ptb']).evaluate(gold)
        print '{0:.3f}'.format(t['parser_score']), scores['gold'], scores['test'], scores['matched']
        print t['seq']
