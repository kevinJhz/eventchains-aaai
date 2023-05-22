#!/bin/bash
# Call from pipeline.sh.

# Make sure that all subprocesses are killed on Ctl+C
trap '{ echo; echo "Exiting on user command"; kill 0; }' SIGINT

mkdir -p $INPUT_TEXT_DIR

echo "Outputting text input to $INPUT_TEXT_DIR"
echo "Processing ${#ALTAVILLA_FILES[@]} input files"
echo

# From the config file, we get an array of input tarballs
for filename in "${ALTAVILLA_FILES[@]}"; do
	echo ">>>>>>>> FILE: $filename, extracting text <<<<<<<<"
	file_basename="$(basename $filename .tar.gz)"
	file_output_dir=$INPUT_TEXT_DIR/$file_basename
	mkdir -p $file_output_dir
	
	# First untar the input tarball
	echo "Extracting files"
	tar -xzf $filename --directory $file_output_dir
	
	echo "Filtering files"
	# Remove METADATA files
	find $file_output_dir -name "METADATA" -exec rm \{\} \;
	# Remove 0th chapters: this is generally short introductory material
	find $file_output_dir -name "*-chap00.txt" -exec rm \{\} \;
	find $file_output_dir -name "*-chap000.txt" -exec rm \{\} \;
	
	# Remove the header from each file
	echo "Removing headers"
	for file in "$( find $file_output_dir -type f )"; do
		sed -i -e '1,/######## BEGIN MAIN TEXT ########/d' $file
	done
	
	# Remove some characters that get in the way of the text
	echo "Simplifying text"
	for file in "$( find $file_output_dir -type f )"; do
		# Underscores are never meaningful for our purposes -- remove them all
		# They often get used for emphasis
		sed -i 's/_/ /g' $file
		# Same goes for stars
		sed -i 's/*/ /g' $file
	done
done
