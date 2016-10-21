LSTM Parse Reranker
----------------------
Overview
~~~~~~
In Preparation

Prerequisites
~~~~~~~~~~~~
`bllipparser <https://pypi.python.org/pypi/bllipparser/2016.9.11>`_

`tensorflow <https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#download-and-setup>`_

Data Preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~
Run the following commands to preprocess data. wsj-train and wsj-dev are training and development files respectively. Each line should have one tree. ::
   
   mkdir emnlp && cd emnlp && git clone https://github.com/cdg720/emnlp2016.git
   ./prepare.sh wsj-train wsj-dev

If you have a license for the 5th Gigaword and want the tri-training data, send me an email with the license at dc65@cs.brown.edu. To preprocess the tri-training data, change lines 39 and 49 in prepare.sh and run prepare.sh.
   
