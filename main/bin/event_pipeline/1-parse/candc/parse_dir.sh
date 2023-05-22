#!/bin/bash
# Parse a whole directory using C&C
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
../../../run_py ../../../../src/main/python/cam/whim/candc/parsedir.py $*
