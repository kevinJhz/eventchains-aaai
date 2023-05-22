#!/bin/bash

if [ ! -d $INPUT_TOKENIZED_DIR ]; then
    echo "Tokenized texts not available in $INPUT_TOKENIZED_DIR"
    if [ ! -f $INPUT_TOKENIZED_ARCHIVE ]; then
        echo "Archived tokenized texts not available in $INPUT_TOKENIZED_ARCHIVE either"
    else
        echo "Unpacking archived tokenized texts from $INPUT_TOKENIZED_ARCHIVE to $INPUT_TOKENIZED_DIR..."
        mkdir -p $INPUT_TOKENIZED_DIR
        cd $INPUT_TOKENIZED_DIR
        tar -xzf $INPUT_TOKENIZED_ARCHIVE
        echo "Tokenized texts now available in $INPUT_TOKENIZED_DIR"
        echo
    fi
fi
