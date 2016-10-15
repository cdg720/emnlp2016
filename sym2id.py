import collections, gzip, sys

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


def train():
  if len(sys.argv) != 2:
    print 'usage: python sym2id.py train.gz'
    sys.exit(0)
    
  vocabs = _build_vocab(sys.argv[1])
  for word, i in vocabs.iteritems():
    print word, i

    
if __name__ == '__main__':
  train()
