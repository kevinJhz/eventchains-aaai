#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

$SCRIPT_DIR/../../../run_py -m cam.whim.entity_narrative.eval.multiple_choice.significance \
    /local/scratch/mtw29/choice_output/gigaword/ \
    $*