#!/bin/bash
# Call from pipeline.sh.

# Tokenized text should now be in $INPUT_TOKENIZED_DIR
# Archive it for future use in the final dir
rm -rf $INPUT_TOKENIZED_ARCHIVE

echo "Archiving tokenized text to $INPUT_TOKENIZED_ARCHIVE"
cd $INPUT_TOKENIZED_DIR
tar -czf $INPUT_TOKENIZED_ARCHIVE *
