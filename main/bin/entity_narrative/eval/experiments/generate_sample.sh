#!/bin/bash

DEV=""

working_dir="/local/scratch/$USER/cloze_output"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ -n "$DEV" ]; then
    echo "Generating dev sample"
    docs="/anfs/bigdisc/mtw29/chains/gigaword-nyt/rich_docs/dev/"
    out_dir="/anfs/bigdisc/mtw29/chains/gigaword-nyt/eval/multiple_choice/dev_1k/"
    samples="1000"
else
    echo "Generating test sample"
    docs="/anfs/bigdisc/mtw29/chains/gigaword-nyt/rich_docs/test/"
    out_dir="/anfs/bigdisc/mtw29/chains/gigaword-nyt/eval/multiple_choice/test_10k/"
    samples="10000"
fi

../../../run_py -m cam.whim.entity_narrative.eval.multiple_choice.generate_questions \
    $docs  $out_dir \
    --unbalanced --tarred \
    --stoplist /anfs/bigdisc/mtw29/chains/gigaword-nyt/rich_docs/stoplist_10.txt \
    --samples $samples \
    $*
