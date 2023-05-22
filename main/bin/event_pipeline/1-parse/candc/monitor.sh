#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
../../../run_py../ ../../../src/main/python/cam/whim/candc/monitor.py $*
