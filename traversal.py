from bllipparser import RerankingParser, Tree
from collections import deque
from process import generate_nbest, open_file, read_vocabs, remove_duplicates
from unk import unkify
import gzip, nltk, sys


def ptb(t, words):
  forms = []
  ptb_recurse(t, words, forms)
  return ' ' + ' '.join(forms) + ' '


def ptb_recurse(t, words, forms):
  forms.append('(' + t.label())
  for child in t: 
    if child.height() == 2:
      if child[0].lower() not in words:
        forms.append(unkify.unkify(child[0]))
      else:
        forms.append(child[0].lower())
    else:
      ptb_recurse(child, words, forms)
  forms.append(')' + t.label())


if __name__ == '__main__':
  if len(sys.argv) != 3 and len(sys.argv) != 4:
    print 'usage: python traversal.py vocabs.gz gold.gz [nbest.gz]'
    sys.exit(0)

  words = read_vocabs(sys.argv[1])
  if len(sys.argv) == 4:
    for line in open_file(sys.argv[2]):
      t = nltk.Tree.fromstring(line[:-1])[0]
      print ptb(t, words)
  else:
    rrp = RerankingParser()
    rrp.load_parser_model('/home/dc65/.local/share/bllipparser/OntoNotes-WSJ/parser')
    for gold, nbest in zip(open_file(sys.argv[2]), generate_nbest(open_file(sys.argv[3]))):
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
