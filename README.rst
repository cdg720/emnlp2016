LSTM Parse Reranker
-------------------
Overview
~~~~~~~~
Code and models used in our paper:

* Do Kook Choe and Eugene Charniak. "`Parsing as Language Modeling <http://cs.brown.edu/people/dc65/papers/emnlp16.pdf>`_." Proceedings of the Conference on `Empirical Methods in Natural Language Processing (EMNLP 2016), 2016

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

Due to stochasticity, runs of training may produce models with slightly different results from what we report in the paper. We have retrained a few models after the submission of our paper and we report F1s and numbers of epochs of several training runs:
   
+--------+-----+-----+-----+-----+-----+-----+
|  wsj   |Paper|  1  |  2  |   3 |    4|   5 |
+--------+-----+-----+-----+-----+-----+-----+
|F1 (dev)|91.62|91.40|91.62|91.59|91.51|91.47|
+--------+-----+-----+-----+-----+-----+-----+
|# epochs| 37  | 47  |  38 | 29  | 45  | 43  |
+--------+-----+-----+-----+-----+-----+-----+

+--------+-----+-----+-----+-----+-----+-----+
|  semi  |Paper|  1  |  2  |   3 |    4|   5 |
+--------+-----+-----+-----+-----+-----+-----+
|F1 (dev)|92.46|92.33|92.52|92.45|92.42|92.42|
+--------+-----+-----+-----+-----+-----+-----+
|# epochs| 26  | 13  |  21 | 31  | 21  | 26  |
+--------+-----+-----+-----+-----+-----+-----+

Note that the F1s reported here are about 0.05-0.1 lower than they actually are. Between training epochs, we evaluate models with batch size 200, which allows faster but less accurate evaluation. Evaluating with batch size 10 recovers full performance of models.


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
