#!/bin/bash
# Call from pipeline.sh.

# Text should now be in $INPUT_TEXT_DIR
# Archive it for future use in the final dir
rm -rf $INPUT_TEXT_ARCHIVE

echo "Archiving extracted text to $INPUT_TEXT_ARCHIVE"
cd $INPUT_TEXT_DIR
tar -czf $INPUT_TEXT_ARCHIVE *
