#!/usr/bin/env bash
#
# Runs the document preprocessing tools, including tokenization 
#  and sentence splitting. Previously this used the Stanford 
#  document preprocessor, but now it uses OpenNLP, which does better 
#  sentence splitting (and should be faster).
#

if [ $# != 2 ]
then
    echo "Usage: preprocess.sh <input-file> <output-file>"
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

SENT_MODEL=$SCRIPT_DIR/../../../../models/en-sent.bin
TOK_MODEL=$SCRIPT_DIR/../../../../models/en-token.bin

input_file=$1
output_file=$2

# Make sure the directory we're outputting to exists
output_dir=$( dirname $output_file )
if [ ! -d $output_dir ]
then
    mkdir -p $output_dir
fi

# If there are lines that should continue sentences without spaces at 
#  the end, we get words run together: put a space at the end of every line
# For some reason, dashes get replaced by underscores - map them back
# Use awk to filter out long lines: any line with more than 60 words is ignored
sed '{:q;N;s/\n/ /g;t q}' <$input_file | \
    ../../../run opennlp.tools.cmdline.CLI SentenceDetector $SENT_MODEL 2>/dev/null | \
    ../../../run opennlp.tools.cmdline.CLI TokenizerME $TOK_MODEL 2>/dev/null | \
    sed -e 's/_/-/g' | \
    awk '{ if (NF <= 60) print; }' >$output_file
