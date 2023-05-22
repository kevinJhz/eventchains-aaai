#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

usage="Usage: config_train.sh <config-name>"
if [ $# -lt 1 ]; then
	echo "Missing config file name"
	echo "$usage"
	exit 1
fi

config_name=$1
shift

if [ ! -f $config_name ]; then
    echo "Config file $config_name does not exist"
	echo "$usage"
	exit 1
fi

# Load configuration settings
. $config_name

$SCRIPT_DIR/../../run_py -m cam.whim.entity_narrative.models.base.train \
    $MODEL_TYPE $INPUT $MODEL_NAME $TRAINING_OPTIONS $*
