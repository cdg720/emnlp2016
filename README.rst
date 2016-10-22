LSTM Parse Reranker
-------------------
Overview
~~~~~~~~
Code and models used in `Parsing as Language Modeling <http://cs.brown.edu/people/dc65/papers/emnlp16.pdf>`_.

Prerequisites
~~~~~~~~~~~~~
`bllipparser <https://pypi.python.org/pypi/bllipparser/2016.9.11>`_

`tensorflow <https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#download-and-setup>`_

Data Preprocessing
~~~~~~~~~~~~~~~~~~
Run the following commands to preprocess data.::
   
   mkdir emnlp && cd emnlp && git clone https://github.com/cdg720/emnlp2016.git
   ./prepare.sh wsj-train wsj-dev

wsj-train and wsj-dev should have one tree per line::

  Tree 1
  Tree 2
  ...

If you have a license for the 5th Gigaword and want the tri-training data we use in the paper, send me an email with the license at dc65@cs.brown.edu. To preprocess the tri-training data, change lines 38 and 48 in prepare.sh appropriately and run prepare.sh.

Training and reranking code is based on ptb_word_lm.py and reader.py of the tensorflow RNN tutorial.

Training
~~~~~~~~
::
   
   mkdir -p models/wsj && python train.py --data_path=wsj --model_path=models/wsj/model

::
      
   mkdir -p models/semi && python tri_train.py --data_path=semi --model_path=models/semi/model

Due to stochasticity, training may produce models with slightly different results from what we report in the paper.
   
+--------+-----+-----+-----+-----+-----+-----+
|  wsj   |Paper|  1  |  2  |   3 |    4|   5 |
+--------+-----+-----+-----+-----+-----+-----+
|F1 (dev)|91.62|91.40|91.62|91.59|91.51|91.47|
+--------+-----+-----+-----+-----+-----+-----+
|# epochs| 37  | 47  |  38 | 29  | 45  | 43  |
+--------+-----+-----+-----+-----+-----+-----+

Reranking
~~~~~~~~~
::
   
   python rerank.py --data_path=data --model_path=models/wsj/model --nbest_path=nbest

nbest has the following format::

  n1
  Tree 1.1
  Tree 1.2
  ...
  Tree 1.n1

  n2
  Tree 2.1
  Tree 2.2
  ...
  Tree 2.n2
  
  ...

Models
~~~~~~
wsj model (coming soon)

`semi-supervised model <http://cs.brown.edu/~dc65/models/semi.tgz>`_
