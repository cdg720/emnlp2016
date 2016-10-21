from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from random import shuffle

from utils import nbest_iterator, ptb_iterator

import itertools, sys, time
import cPickle as pickle
import numpy as np
import tensorflow as tf

import reader

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("data_path", None, "data_path")
flags.DEFINE_float('init_scale', 0.05, 'init_scale')
flags.DEFINE_float('learning_rate', 0.25, 'learning_rate')
flags.DEFINE_float('max_grad_norm', 20, 'max_grad_norm')
flags.DEFINE_integer('num_layers', 3, 'num_layers')
flags.DEFINE_integer('num_steps', 50, 'num_steps')
flags.DEFINE_integer('hidden_size', 1500, 'hidden_size')
flags.DEFINE_integer('max_epoch', 14, 'max_epoch')
flags.DEFINE_integer('max_max_epoch', 50, 'max_max_epoch')
flags.DEFINE_float('keep_prob', 0.3, 'keep_prob')
flags.DEFINE_float('lr_decay', 0.85, 'lr_decay')
flags.DEFINE_integer('batch_size', 20, 'batch_size')
flags.DEFINE_string('model_path', None, 'model_path')

FLAGS = flags.FLAGS

class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size

    self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=1.0,
                                             state_is_tuple=True)
    if is_training and config.keep_prob < 1:
      lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
          lstm_cell, output_keep_prob=config.keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers,
                                       state_is_tuple=True)

    self._initial_state = cell.zero_state(batch_size, tf.float32)

    with tf.device("/cpu:0"):
      embedding = tf.get_variable("embedding", [vocab_size, size])
      inputs = tf.nn.embedding_lookup(embedding, self._input_data)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    inputs = [tf.squeeze(input_, [1])
              for input_ in tf.split(1, num_steps, inputs)]
    outputs, state = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)

    output = tf.reshape(tf.concat(1, outputs), [-1, size])
    softmax_w = tf.get_variable("softmax_w", [size, vocab_size])
    softmax_b = tf.get_variable("softmax_b", [vocab_size])
    logits = tf.matmul(output, softmax_w) + softmax_b
    loss = tf.nn.seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(self._targets, [-1])],
        [tf.ones([batch_size * num_steps])])
    cost = tf.reduce_sum(loss) / batch_size
    self._cost = loss
    self._final_state = state

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self.lr)
    self._train_op = optimizer.apply_gradients(zip(grads, tvars))

  def assign_lr(self, session, lr_value):
    session.run(tf.assign(self.lr, lr_value))

  @property
  def input_data(self):
    return self._input_data

  @property
  def targets(self):
    return self._targets

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20


def run_epoch(session, m, data, eval_op, verbose=False):
  """Runs the model on the given data."""
  epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = []
  for c, h in m.initial_state: # initial_state: ((c1, m1), (c2, m2))
    state.append((c.eval(), h.eval()))
  for step, (x, y) in enumerate(ptb_iterator(data, m.batch_size, m.num_steps)):
    fetches = []
    fetches.append(m.cost)
    fetches.append(eval_op)
    for c, h in m.final_state: # final_state: ((c1, m1), (c2, m2))
      fetches.append(c)
      fetches.append(h)
    feed_dict = {}
    feed_dict[m.input_data] = x
    feed_dict[m.targets] = y
    for i, (c, h) in enumerate(m.initial_state):
      feed_dict[c], feed_dict[h] = state[i]
    res = session.run(fetches, feed_dict)
    cost = res[0]
    state_flat = res[2:] # [c1, m1, c2, m2]
    state = [state_flat[i:i+2] for i in range(0, len(state_flat), 2)]
    costs += np.sum(cost) / m.batch_size
    iters += m.num_steps

    if verbose and step % (epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / epoch_size, np.exp(costs / iters),
             iters * m.batch_size / (time.time() - start_time)))

  return np.exp(costs / iters)


def run_epoch2(session, m, nbest, eval_op, eos, verbose=False):
  """Runs the model on the given data."""
  counts = []
  loss = []
  prev = (-1, -1)
  for pair in nbest['idx2tree']:
    if pair[0] != prev[0]:
      counts.append([0])
      loss.append([0.])
    elif pair[1] == prev[1] + 1:
      counts[-1].append(0)
      loss[-1].append(0.)
    counts[-1][-1] += 1
    prev = pair
  data = nbest['data']    
  epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = []
  for c, h in m.initial_state: # initial_state: ((c1, m1), (c2, m2))
    state.append((c.eval(), h.eval()))
  for step, (x, y, z) in enumerate(
          nbest_iterator(data, m.batch_size, m.num_steps,
                         nbest['idx2tree'], eos)):
    fetches = []
    fetches.append(m.cost)
    fetches.append(eval_op)
    for c, h in m.final_state: # final_state: ((c1, m1), (c2, m2))
      fetches.append(c)
      fetches.append(h)
    feed_dict = {}
    feed_dict[m.input_data] = x
    feed_dict[m.targets] = y
    for i, (c, h) in enumerate(m.initial_state):
      feed_dict[c], feed_dict[h] = state[i]
    res = session.run(fetches, feed_dict)
    cost = res[0]
    state_flat = res[2:] # [c1, m1, c2, m2]
    state = [state_flat[i:i+2] for i in range(0, len(state_flat), 2)]
    costs += np.sum(cost) / m.batch_size
    iters += m.num_steps

    cost = cost.reshape((m.batch_size, m.num_steps))
    for idx, val in np.ndenumerate(cost):
      tree_idx = z[idx[0]][idx[1]]
      if tree_idx[0] == -1: # dummy
        continue
      counts[tree_idx[0]][tree_idx[1]] -= 1
      loss[tree_idx[0]][tree_idx[1]] += cost[idx[0]][idx[1]]
              
    if verbose and step % (epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / epoch_size, np.exp(costs / iters),
             iters * m.batch_size / (time.time() - start_time)))

  scores = nbest['scores']
  num = 0
  gold, test, matched = 0, 0, 0
  bad = []
  for i in xrange(len(scores)):
    good = True
    ag = 0
    min_val = 10000000
    for j in xrange(len(scores[i])):
      if counts[i][j] != 0:
        bad.append(i)
        good = False
        break
      if loss[i][j] < min_val:
        min_val = loss[i][j]
        ag = j
    if good:
      num += 1      
      gold += scores[i][ag]['gold']
      test += scores[i][ag]['test']
      matched += scores[i][ag]['matched']
  if bad:
    print('bad: %s' % ', '.join([str(x) for x in bad]))
  return 200. * matched / (gold + test), num


def chop(data, eos):
  new_data = []
  sent = []
  for w in data:
    sent.append(w)
    if w == eos:
      new_data.append(sent)
      sent = []
  return new_data
  

def train():
  print('data_path: %s' % FLAGS.data_path)
  raw_data = reader.ptb_raw_data(FLAGS.data_path)
  train_data, valid_data, valid_nbest_data, vocab = raw_data
  train_data = chop(train_data, vocab['<eos>'])
  
  config = MediumConfig()
  config.init_scale = FLAGS.init_scale
  config.learning_rate = FLAGS.learning_rate
  config.max_grad_norm = FLAGS.max_grad_norm
  config.num_layers = FLAGS.num_layers
  config.num_steps = FLAGS.num_steps
  config.hidden_size = FLAGS.hidden_size
  config.max_epoch = FLAGS.max_epoch
  config.max_max_epoch = FLAGS.max_max_epoch
  config.keep_prob = FLAGS.keep_prob
  config.lr_decay = FLAGS.lr_decay
  config.batch_size = FLAGS.batch_size
  config.vocab_size = len(vocab)
  print('init_scale: %.2f' % config.init_scale)
  print('learning_rate: %.2f' % config.learning_rate)
  print('max_grad_norm: %.2f' % config.max_grad_norm)
  print('num_layers: %d' % config.num_layers)
  print('num_steps: %d' % config.num_steps)
  print('hidden_size: %d' % config.hidden_size)
  print('max_epoch: %d' % config.max_epoch)
  print('max_max_epoch: %d' % config.max_max_epoch)
  print('keep_prob: %.2f' % config.keep_prob)
  print('lr_decay: %.2f' % config.lr_decay)
  print('batch_size: %d' % config.batch_size)
  print('vocab_size: %d' % config.vocab_size)
  sys.stdout.flush()
  
  eval_config = MediumConfig()
  eval_config.init_scale = FLAGS.init_scale
  eval_config.learning_rate = FLAGS.learning_rate
  eval_config.max_grad_norm = FLAGS.max_grad_norm
  eval_config.num_layers = FLAGS.num_layers
  eval_config.num_steps = FLAGS.num_steps
  eval_config.hidden_size = FLAGS.hidden_size
  eval_config.max_epoch = FLAGS.max_epoch
  eval_config.max_max_epoch = FLAGS.max_max_epoch
  eval_config.keep_prob = FLAGS.keep_prob
  eval_config.lr_decay = FLAGS.lr_decay
  eval_config.batch_size = 200
  eval_config.vocab_size = len(vocab)

  prev = 0
  with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
      m = PTBModel(is_training=True, config=config)
    with tf.variable_scope("model", reuse=True, initializer=initializer):
      mvalid = PTBModel(is_training=False, config=eval_config)

    tf.initialize_all_variables().run()
    if FLAGS.model_path:
      saver = tf.train.Saver()

    for i in range(config.max_max_epoch):
      shuffle(train_data)
      shuffled_data = list(itertools.chain(*train_data))
      
      start_time = time.time()
      lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
      m.assign_lr(session, config.learning_rate * lr_decay)

      print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
      train_perplexity = run_epoch(session, m, shuffled_data, m.train_op,
                                   verbose=True)
      print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
      valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op())
      print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
      valid_f1, num = run_epoch2(session, mvalid, valid_nbest_data,
                                 tf.no_op(), vocab['<eos>'])
      print("Epoch: %d Valid F1: %.2f (%d trees)" % (i + 1, valid_f1, num))
      print('It took %.2f seconds' % (time.time() - start_time))
      if prev < valid_f1:
        prev = valid_f1
        if FLAGS.model_path:
          print('Save a model to %s' % FLAGS.model_path)
          saver.save(session, FLAGS.model_path)
          pickle.dump(eval_config, open(FLAGS.model_path + '.config', 'wb'))
      sys.stdout.flush()


def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  print(' '.join(sys.argv))
  train()
    

if __name__ == "__main__":
  tf.app.run()
