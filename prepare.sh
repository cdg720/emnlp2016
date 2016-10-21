#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "echo ./prepare.sh wsj-train.gz wsj-dev.gz"
    exit
fi

# One tree per line.
TRAIN=$1
DEV=$2

mkdir data
# Remove function tags.
if [[ "$TRAIN" == *.gz ]]
then
   zcat $TRAIN | python strip_function_tags.py | gzip > data/x.gz
   zcat $DEV | python strip_function_tags.py | gzip > data/y.gz
else
   cat $TRAIN | python strip_function_tags.py | gzip > data/x.gz
   cat $DEV | python strip_function_tags.py | gzip > data/y.gz
fi

# Download Charniak parser.
python -mbllipparser.ModelFetcher -i WSJ-PTB3 -d data
# ln -s /pro/dpg/dc65/reranker/test/all/ -d data/WSJ-PTB3

# Generate nbest parses with Charniak parser. On a modern processer, parsing
# section 24 takes about 5 minutes. 
zcat data/y.gz | python nbest_parse.py | gzip > data/z.gz

# Create a vocab file.
python create_vocab.py data/x.gz 9 | gzip > data/vocab.gz

# Preprocess train, dev and dev_nbest files.
python traversal.py data/vocab.gz data/x.gz | gzip > data/train.gz
python traversal.py data/vocab.gz data/y.gz | gzip > data/dev.gz
python traversal.py data/vocab.gz data/y.gz data/z.gz | gzip > data/dev_nbest.gz

# Remove unnecessary data.
rm data/[xyz].gz
