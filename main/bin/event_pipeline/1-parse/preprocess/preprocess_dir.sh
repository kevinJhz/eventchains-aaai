#!/usr/bin/env bash
#
# Runs the document preprocessing tools, including tokenization 
#  and sentence splitting, a directory at a time.
#

if [ $# != 2 ]
then
    echo "Usage: preprocess.sh <input-dir> <output-dir>"
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

SENT_MODEL=$SCRIPT_DIR/../../../../models/en-sent.bin
TOK_MODEL=$SCRIPT_DIR/../../../../models/en-token.bin

input_dir=$1
output_dir=$2

# Make sure the directory we're outputting to exists
if [ ! -d $output_dir ]; then
    mkdir -p $output_dir
fi

echo "Processing $(find $input_dir -type f | wc -l) files" 1>&2
for filename in $( cd $input_dir; find * -type f ); do
    # Make sure the output file's directory (which could be a subdir of 
    #  output_dir) exists
    mkdir -p $(dirname $output_dir/$filename)
    
    # Change output file
    echo; echo "%% OUTPUT: $output_dir/$filename"; echo
    
    # Output the contents of the file
    # Here we insert any special rules we want for tokenization that the tokenizer doesn't handle
    # Some texts make heavy use of m-dashes represented as "--", with no spaces: space them out
    sed 's/\([^ -]\)--\([^ -]\)/\1 -- \2/g' <$input_dir/$filename
    
    echo                # File might not end in a newline
done | ../../../run cam.whim.opennlp.Tokenize $SENT_MODEL $TOK_MODEL

echo "Filtering output" 1>&2
for filename in $( find $output_dir -type f ); do
    filebase=$(basename $filename)
    # For some reason, dashes get replaced by underscores - map them back
    sed -i -e 's/_/-/g' $filename
    # Use awk to filter out long lines: any line with more than 60 words is ignored
    awk '{ if (NF <= 60) print; }' <$filename >/tmp/cutlong.$filebase.tmp
    mv /tmp/cutlong.$filebase.tmp $filename
done
