LSTM Parse Reranker
----------------------
Overview
~~~~~~
Code and models used in `Parsing as Language Modeling <http://cs.brown.edu/people/dc65/papers/emnlp16.pdf>`_.

Prerequisites
~~~~~~~~~~~~
`bllipparser <https://pypi.python.org/pypi/bllipparser/2016.9.11>`_

`tensorflow <https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#download-and-setup>`_

Data Preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~
Run the following commands to preprocess data.::
   
   mkdir emnlp && cd emnlp && git clone https://github.com/cdg720/emnlp2016.git
   ./prepare.sh wsj-train wsj-dev

wsj-train and wsj-dev should have one tree per line::

  Tree 1
  Tree 2
  ...

If you have a license for the 5th Gigaword and want the tri-training data we use in the paper, send me an email with the license at dc65@cs.brown.edu. To preprocess the tri-training data, change lines 39 and 49 in prepare.sh appropriately and run prepare.sh.
   
