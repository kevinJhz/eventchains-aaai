#!/bin/bash
# Call from pipeline.sh.

# If the text dir doesn't exist, try getting it from an archive
if [ ! -d $INPUT_TEXT_DIR ]; then
	echo "Input text dir does not exist: $INPUT_TEXT_DIR"
	if [ -f $INPUT_TEXTARCHIVE ]; then
		echo "Getting input texts from archive"
		mkdir -p $INPUT_TEXT_DIR
		cd $INPUT_TEXT_DIR
		tar -xzf $INPUT_TEXT_ARCHIVE
	else
		echo "Input text dir not found and no archive available"
		exit 1
	fi
fi

# Make sure output dir exists
rm -rf $INPUT_TOKENIZED_DIR
mkdir -p $INPUT_TOKENIZED_DIR

echo "Outputting to $INPUT_TOKENIZED_DIR"
echo "Processing $PROCESSES in parallel"

# Define a function to run in parallel
preprocess() {
	dir=$1
	
	# Preprocess whole directory
	echo "$dir: starting"
	$PREPROCESS_DIR_CMD $INPUT_TEXT_DIR/$dir $INPUT_TOKENIZED_DIR/$dir #2>/dev/null
	
	# Check that all the files in the input dir got a file in the output dir
	input_files=$( find $INPUT_TEXT_DIR/$dir -type f | wc -l )
	output_files=$( find $INPUT_TOKENIZED_DIR/$dir -type f | wc -l )
	if [ "$input_files" == "$output_files" ]; then
		# Preprocessed all files in this directory: remove it
		rm -rf $INPUT_TEXT_DIR/$dir
	else
		echo "Wrong number of files in output dir, $INPUT_TOKENIZED_DIR/$dir: not removing input"
	fi
	echo "$dir: finished"
	return 0
}
export -f preprocess
export INPUT_TEXT_DIR INPUT_TOKENIZED_DIR
PREPROCESS_DIR_CMD=$SCRIPT_DIR/1-parse/preprocess/preprocess_dir.sh
export PREPROCESS_DIR_CMD


# Check whether the text dir has subdirectories
if [ -n "$(find $INPUT_TEXT_DIR -maxdepth 1 -type d)" ]; then
	cd $INPUT_TEXT_DIR; input_dirs=(*); cd - >/dev/null
	echo "Processing ${#input_dirs[@]} input directories"
	echo
	
	# Split into subdirectories of the input dir so we can report progress better
	printf "%s\n" "${input_dirs[@]}" | \
		xargs -n 1 -P $PROCESSES bash -c 'preprocess "$@"' "preprocess()" # This string is arg0 (the script name), arg1 is the input file
	echo "Done"
else
	echo "ERROR: no subdirectories found in input text dir. I've not coded this case -- do it if you want"
fi

# Clean up
if [ -z "$( ls -A $INPUT_TEXT_DIR )" ]; then
	# Processed all files: remove directory
	echo "Removing input dir $INPUT_TEXT_DIR"
	rm -rf $INPUT_TEXT_DIR
else
	echo "Some files remaining in $INPUT_TEXT_DIR: not removing"
fi
