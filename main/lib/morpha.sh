#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR/morph

# Create a pipeline that POS tags using C&C and then uses Morpha to stem
# We need to replace the | separators from C&C with _s
../candc-1.00/bin/pos --model ../../models/candc/pos <&0 | \
	sed 's/|/_/g' | \
	./morpha $*
