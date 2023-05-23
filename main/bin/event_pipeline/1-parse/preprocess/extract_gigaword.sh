#!/bin/bash
# Call from pipeline.sh.

# Make sure that all subprocesses are killed on Ctl+C
trap '{ echo; echo "Exiting on user command"; kill 0; }' SIGINT

# From the config file, we get an array of input tarballs

# Temporary dir for extracting files
EXTRACTOR_TEMP_DIR=$TEMP_DIR/extractor
# Blacklist file for document names to skip
BLACKLIST=$SCRIPT_DIR/1-parse/preprocess/gigaword/gigaword_duplicates.gz

# Make sure output dirs exist
mkdir -p $EXTRACTOR_TEMP_DIR
mkdir -p $INPUT_TEXT_DIR

echo "Outputting tokenized input to $INPUT_TOKENIZED_DIR"
echo "Processing ${#GIGAWORD_FILES[@]} input files"  # 数组变量前面加 #号，表示该数组变量的元素个数，[@]是用来表示数组变量的所有元素
echo "Temporary files in $EXTRACTOR_TEMP_DIR"
echo

cd $SCRIPT_DIR/1-parse/preprocess
# Extract every file in the Gigaword dir
for filename in "${GIGAWORD_FILES[@]}"; do
	echo ">>>>>>>> FILE: $filename, extracting text <<<<<<<<"
	file_basename="$(basename $filename .gz)"
	file_tmp_dir=$EXTRACTOR_TEMP_DIR/$file_basename
	mkdir -p $file_tmp_dir
	file_output_dir=$INPUT_TEXT_DIR/$file_basename

	echo "Building file list"
	# Output the list of files to be extracted to a file
#	echo $(pwd)
#	echo $filename
#	echo $file_output_dir
#	echo $file_tmp_dir
	../../../run_py ./gigaword/gigaword_split.py --list $filename --type story | \
		../../../run_py ../filter_exclude.py $BLACKLIST --silent | \
		../../../run_py ../filter_done.py $file_output_dir --allow-no-output --basenames \
			>$file_tmp_dir/to_parse.txt

	# If the file list is empty, we've done the whole of this input before
	to_do=$(cat $file_tmp_dir/to_parse.txt | wc -l)
	if [ $to_do -eq 0 ]; then
		echo "All files already extracted, see $INPUT_TEXT_DIR"
		continue
		echo
		echo "Deleting tmp dir $file_tmp_dir"
    rm -rf $file_tmp_dir
	fi

	# Split the file into one file per doc
	echo "Getting documents from Gigaword file"
#	echo $INPUT_TEXT_DIR/$file_basename
#	echo $file_tmp_dir/to_parse.txt
#	echo $filename
	./gigaword/gigaword_split.py --type story \
				--output $INPUT_TEXT_DIR/$file_basename \
				--include $file_tmp_dir/to_parse.txt \
				$filename
	echo "Documents have been extracted FROM $filename TO $INPUT_TEXT_DIR"
	echo
	echo "Deleting tmp dir $file_tmp_dir"
  rm -rf $file_tmp_dir
done
