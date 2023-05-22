#!/bin/bash

USAGE="Usage: gigaword_multiple_choice.sh [-h] [-s <model-name-suffix>] <model-type> <model-name> [<options>]"
working_dir="/local/scratch/$USER/cloze_output"
model_name_suffix=""
# Default to using dev set
test_set="/anfs/bigdisc/mtw29/chains/gigaword-nyt/eval/multiple_choice/dev_1k/"
output_base_dir_prefix=""

while getopts "hs:t" opt; do
    case $opt in
    h)
        echo $USAGE
        echo "  -h  Print this message"
        echo "  -s SUFFIX"
        echo "      Add a suffix to the end of model name when storing results"
        echo "  -t  Evaluate on test (instead of dev) set"
        exit 0
        ;;
    s)
        model_name_suffix=$OPTARG
        ;;
    t)
        test_set="/anfs/bigdisc/mtw29/chains/gigaword-nyt/eval/multiple_choice/test_10k/"
        output_base_dir_prefix="test/"
        echo "EVALUATING ON TEST SET: $test_set"
        ;;
    \?)
        echo "Invalid option" >&2
        exit 1
        ;;
    :)
        echo "Option -$opt requires an argument." >&2
        exit 1
        ;;
  esac
done
shift $(($OPTIND-1))

if [ $# -lt 2 ]
then
    echo $USAGE
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

model_type=$1
model_name=$2
model_output_name="$model_name"
if [ -n "$model_name_suffix" ]; then
    model_output_name="$model_name-$model_name_suffix"
    echo "Applying suffix to model name for output: $model_output_name"
fi
shift 2

../../../run_py -m cam.whim.entity_narrative.eval.multiple_choice.predict \
    $model_type $model_name \
    $test_set \
    /local/scratch/mtw29/choice_output/gigaword/$output_base_dir_prefix$model_type/$model_output_name \
    --prepared \
    $*
