from unk import unkify
import gzip, nltk, sys


def df(t, words, lr=True):
  forms = []
  df_recurse(t, words, forms, lr)
  return [words[x] for x in forms + ['<eos>']]


def df_recurse(t, words, forms, lr=True):
  forms.append('(' + t.label())
  mid = False
  for child in t: 
    if mid:
      forms.append('<' + t.label() + '>')
    if child.height() == 2:
      if child[0].lower() not in words:
        forms.append(unkify.unkify(child[0]))
      else:
        forms.append(child[0].lower())
    else:
      df_recurse(child, words, forms, lr)
    mid = True
  forms.append(')' + t.label())


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


def read_vocabs(filename):
  word2id = {}
  for line in gzip.open(filename, 'rb'):
    word, i = line.split()
    word2id[word] = int(i)
  return word2id


def trees2ints(filename, word2id):
  data = []
  for i, line in enumerate(gzip.open(filename, 'rb')):
    data.extend(df(nltk.Tree.fromstring(line)[0], word2id))
    # data.extend(ptb(nltk.Tree.fromstring(line)[0], word2id))
    if (i + 1) % 40000 == 0:
      yield data
      data = []
  if data:
    yield data


if __name__ == '__main__':
  if len(sys.argv) != 3:
    print 'usage: python trees2ints.py word2id.gz data.gz'
    sys.exit(0)

  word2id = read_vocabs(sys.argv[1])
  for data in trees2ints(sys.argv[2], word2id):
    print ' '.join([str(x) for x in data])

  # sanity check
  # word_to_id = _build_vocab(sys.argv[2])
  # print _file_to_word_ids(sys.argv[2], word_to_id)
