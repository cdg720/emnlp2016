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

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, time

import cPickle as pickle
import numpy as np
import tensorflow as tf
from nltk import Tree

import reader2

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "medium",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None, "data_path")
flags.DEFINE_string('model_path', None, 'model_path')
flags.DEFINE_string('nbest_path', None, 'nbest_path')
flags.DEFINE_boolean('nbest', False, 'nbest')

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

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
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

    outputs = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)

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
  vocab_size = 9855

  
def run_epoch(session, m, nbest, eval_op, eos, verbose=False):
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
          reader2.ptb_iterator(data, m.batch_size, m.num_steps,
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

  trees = nbest['trees']
  bad = []
  num_words = 0
  for i in xrange(len(trees)):
    good = True
    ag = 0
    min_val = float('inf')
    for j in xrange(len(trees[i])):
      if counts[i][j] != 0:
        bad.append(i)
        good = False
        break
      if loss[i][j] < min_val:
        min_val = loss[i][j]
        ag = j
    if good:
      if FLAGS.nbest:
        print(len(trees[i]))
        for j in xrange(len(trees[i])):
          print(loss[i][j])
          print(trees[i][j])
        print()
      else:
        print(trees[i][ag])
  if bad:
    print('bad: %s' % ', '.join([str(x) for x in bad]))


def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  elif FLAGS.model == "medium":
    return MediumConfig()
  elif FLAGS.model == "large":
    return LargeConfig()
  elif FLAGS.model == "test":
    return TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)


def test():
  config = pickle.load(open(FLAGS.model_path + '.config', 'rb'))
  config.batch_size = 10
  test_nbest_data, vocab = reader2.ptb_raw_data(FLAGS.data_path,
                                                 FLAGS.nbest_path)
  with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
      m = PTBModel(is_training=False, config=config)

    saver = tf.train.Saver()
    saver.restore(session, FLAGS.model_path)
    run_epoch(session, m, test_nbest_data, tf.no_op(), vocab['<eos>'])

    
def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")
  if not FLAGS.nbest_path:
    raise ValueError("Must set --nbest_path to nbest data")  
  if not FLAGS.model_path:
    raise ValueError("Must set --model_path to model")
  test()
    

if __name__ == "__main__":
  tf.app.run()
