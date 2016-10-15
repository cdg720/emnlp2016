import gzip, nltk, sys


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


def ptb(t, words):
  forms = []
  ptb_recurse(t, words, forms)
  return [words[x] for x in forms + ['<eos>']]


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


def read_vocab(filename):
  word2id = {}
  for line in gzip.open(filename, 'rb'):
    word, i = line.split()
    word2id[word] = int(i)
  return word2id


def integerize(filename, word2id):
  data = []
  for i, line in enumerate(gzip.open(filename, 'rb')):
    data.extend(df(nltk.Tree.fromstring(line)[0], word2id))
    if (i + 1) % 40000 == 0:
      yield data
      data = []
  if data:
    yield data


if __name__ == '__main__':
  if len(sys.argv) != 3:
    print 'usage: python integerize.py word2id.gz data.gz'
    sys.exit(0)

  word2id = read_vocab(sys.argv[1])
  for data in integerize(sys.argv[2], word2id):
    print ' '.join([str(x) for x in data])
