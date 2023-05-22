#!/bin/bash
# Main pipeline script to call all steps of the pipeline
# This calls all other stages of the pipeline in subdirectories and 
#  reads config from a training config file.

# Make sure that all subprocesses are killed on Ctl+C
trap '{ echo; echo "Exiting on user command"; kill 0; }' SIGINT

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

usage="Usage: pipeline.sh <config-name> <pipeline-stage>"
stages="Possible stages: 1 (extract texts), 1b (compress texts, included in 1), 2 (tokenize), 2b (compress tokenized texts, included in 2), 3 (OpenNLP parse), 4 (C&C parse), 5 (coref), 6 (event extraction), 7 (counts), malt3 (OpenNLP postag), malt4 (Malt parse)"
if [ $# -eq 0 ]; then
	echo "Missing pipeline config name"
	echo "$usage"
	available_configs=$(cd config; ls | grep -v local)
	echo
	echo "Available pipelines: "
	echo "$available_configs"
	exit 1
elif [ $# -lt 2 ]; then
	echo "Missing pipeline stage"
	echo "$usage"
	echo "$stages"
	exit 1
fi

config_name=$1
shift
pipeline_stage=$1
shift


# Initiatization
declare -A SHARE_STAGE
declare -A COUNTS_DIRS
# Defaults
COUNTS_CUTOFF=0
CHAIN_EXTRACTION_ARGS=


# Load local config
. config/local

# Load the pipeline config
if [ ! -f config/$config_name ]; then
	echo "No pipeline config named $config_name"
	exit 1
fi
. config/$config_name

#### Process config ####
# The config file gives us a load of variables
# Prepare some derivatives used by the pipeline
WORKING_DIR=$LOCAL_WORKING_DIR/$PIPELINE_NAME
FINAL_DIR=$LOCAL_FINAL_DIR/$PIPELINE_NAME

TEMP_DIR=$WORKING_DIR/tmp
mkdir -p $TEMP_DIR
mkdir -p $FINAL_DIR

if [ -z "$HUMAN_READABLE_NAME" ]; then HUMAN_READABLE_NAME=$PIPELINE_NAME; fi

# Stage 1
if [ -n "${SHARE_STAGE[1]}" ]; then
	# Use another pipeline's dir
	INPUT_TEXT_DIR=$LOCAL_WORKING_DIR/${SHARE_STAGE[1]}/input/text
	INPUT_TEXT_ARCHIVE=$LOCAL_FINAL_DIR/${SHARE_STAGE[1]}/text.tar.gz
else
	INPUT_TEXT_DIR=$WORKING_DIR/input/text
	INPUT_TEXT_ARCHIVE=$FINAL_DIR/text.tar.gz
fi
# Stage 2
if [ -n "${SHARE_STAGE[2]}" ]; then
	# Use another pipeline's dir
	INPUT_TOKENIZED_DIR=$LOCAL_WORKING_DIR/${SHARE_STAGE[2]}/input/tokenized
	INPUT_TOKENIZED_ARCHIVE=$LOCAL_FINAL_DIR/${SHARE_STAGE[2]}/tokenized.tar.gz
else
	INPUT_TOKENIZED_DIR=$WORKING_DIR/input/tokenized
	INPUT_TOKENIZED_ARCHIVE=$FINAL_DIR/tokenized.tar.gz
fi
# Stage 3
if [ -n "${SHARE_STAGE[3]}" ]; then
	TREES_DIR=$LOCAL_FINAL_DIR/${SHARE_STAGE[3]}/opennlp/trees
else
	TREES_DIR=$FINAL_DIR/opennlp/trees
fi
# Stage 4
if [ -n "${SHARE_STAGE[4]}" ]; then
	DEPS_DIR=$LOCAL_FINAL_DIR/${SHARE_STAGE[4]}/candc/deps
	TAGS_DIR=$LOCAL_FINAL_DIR/${SHARE_STAGE[4]}/candc/tags
else
	DEPS_DIR=$FINAL_DIR/candc/deps
	TAGS_DIR=$FINAL_DIR/candc/tags
fi
# Stage 5
if [ -n "${SHARE_STAGE[5]}" ]; then
	COREF_DIR=$LOCAL_FINAL_DIR/${SHARE_STAGE[5]}/coref
else
	COREF_DIR=$FINAL_DIR/coref
fi

MODEL_DIR=$SCRIPT_DIR/../models/eventchains/models
POSTAGS_DIR=$FINAL_DIR/opennlp/pos

# Allow a pipeline to specify a custom counts script
if [ -z "$COUNTS_SCRIPT" ]; then
	COUNTS_SCRIPT=$SCRIPT_DIR/4-count/counts.sh
fi

if [ -z "$POSTPROCESSED_CHAINS_FILE" ]; then POSTPROCESSED_CHAINS_FILE=$CHAINS_FILE; fi
######

echo "#########################"
echo "# Event chains pipeline #"
echo "#   Stage $pipeline_stage             #"
echo "#########################"
echo
echo "Pipeline:  $PIPELINE_NAME"
echo "Processes: $PROCESSES"

# Check this stage is not supposed to be shared with another pipeline:
# if it is, don't allow it to be run for this pipeline
if [ "$pipeline_stage" == "1b" ]; then major_pipeline_stage="1"
elif [ "$pipeline_stage" == "2b" ]; then major_pipeline_stage="2"
else major_pipeline_stage=$pipeline_stage
fi

if [ -n "${SHARE_STAGE[$major_pipeline_stage]}" ]; then
	echo; echo "Pipeline stage $major_pipeline_stage is shared with the pipeline \"${SHARE_STAGE[$major_pipeline_stage]}\""
	echo "Run the stage for that pipeline and its output will be used by this one:"
	echo "  ./pipeline ${SHARE_STAGE[$major_pipeline_stage]} $pipeline_stage"
	exit 1
fi


case "$pipeline_stage" in
	"1")
		echo "Stage 1:   extract texts"
		echo
		# Run the input extraction script given in the config file
		# This should put the input texts in $INPUT_TEXT_DIR
		cd $SCRIPT_DIR/1-parse
		. $SCRIPT_DIR/$INPUT_EXTRACTOR
		
		cd $SCRIPT_DIR/1-parse/preprocess
		. finalize_extraction.sh
		echo; echo "Pipeline stage 1 complete: texts in $INPUT_TEXT_DIR"
		;;
	"1b")
		echo "Stage 1b:  compress texts"
		echo
		echo "Note that this is also performed by stage 1, so you don't need to run it after fully running stage 1"
		cd $SCRIPT_DIR/1-parse/preprocess
		. finalize_extraction.sh
		;;
	"2")
		echo "Stage 2:   preprocess texts (tokenize, etc)"
		echo
		cd $SCRIPT_DIR/1-parse/preprocess
		. prepare_text.sh
		
		cd $SCRIPT_DIR/1-parse/preprocess
		. finalize_preparation.sh
		echo; echo "Pipeline stage 2 complete: tokenized texts in $INPUT_TOKENIZED_DIR"
		;;
	"2b")
		echo "Stage 2b:  compress tokenized texts"
		echo
		echo "Note that this is also performed by stage 2, so you don't need to run it after fully running stage 2"
		echo 
		echo "NB: Currently this won't be unarchived automatically if the tokenized files aren't available, but it's at least there so you can do it yourself"
		cd $SCRIPT_DIR/1-parse/preprocess
		. finalize_preparation.sh
		echo "Tokenized archive: $INPUT_TOKENIZED_ARCHIVE"
		;;
	"3")
		echo "Stage 3:   parse with OpenNLP"
		echo
		cd $SCRIPT_DIR/1-parse/preprocess
		. unpack_tokenized.sh
		cd $SCRIPT_DIR/1-parse/opennlp
		. parse.sh
		echo; echo "Pipeline stage 3 complete: parser output in $TREES_DIR"
		;;
	"4")
		echo "Stage 4:   parse with C&C"
		echo
		cd $SCRIPT_DIR/1-parse/preprocess
		. unpack_tokenized.sh
		cd $SCRIPT_DIR/1-parse/candc
		. parse.sh
		echo; echo "Pipeline stage 4 complete: parser output in $DEPS_DIR and $TAGS_DIR"
		;;
	"5")
		echo "Stage 5:   coreference resolution"
		echo
		cd $SCRIPT_DIR/2-coref
		. coref.sh
		echo; echo "Pipeline stage 5 complete: chains output in $COREF_DIR"
		;;
	*)
		echo "Unknown pipeline stage: $pipeline_stage"
		echo "$stages"
		exit 1
esac
