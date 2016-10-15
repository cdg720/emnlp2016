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


def train_only():
  if len(sys.argv) != 2:
    print 'usage: python word2id.py train.gz'
    sys.exit(0)
    
  vocabs = _build_vocab(sys.argv[1])
  for word, i in vocabs.iteritems():
    print word, i


def train_and_extra():
  if len(sys.argv) != 4:
    print 'usage: python word2id.py nonwords.gz words.gz num'
    sys.exit(0)

  num = int(sys.argv[3])
  i = 0
  for line in gzip.open(sys.argv[1], 'rb'):
    print line.split()[0], i
    i += 1

  for line in gzip.open(sys.argv[2], 'rb'):
    print line.split()[0], i
    i += 1
    if i == num:
      break

    
if __name__ == '__main__':
  # train_and_extra()
  train_only()
