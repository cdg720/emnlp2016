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
Run the following commands to process data. wsj-train and wsj-dev are training and development files. Each line should have one tree. ::
   
   mkdir emnlp && cd emnlp && git clone https://github.com/cdg720/emnlp2016.git
   ./prepare.sh wsj-train wsj-dev


