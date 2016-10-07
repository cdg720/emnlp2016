#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "echo ./prepare.sh wsj-train.gz wsj-dev.gz"
    exit
fi

TRAIN=$1
DEV=$2

mkdir data
zcat TRAIN | gzip > data/x.gz
zcat DEV | gzip > data/y.gz

python -mbllipparser.ModelFetcher -i WSJ-PTB3 .
