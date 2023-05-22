#!/bin/bash
# POS-tag the whole input data with the OpenNLP parser.
# Call from pipeline.sh.

# Check input dir exists
if [ ! -d "$INPUT_TOKENIZED_DIR" ]
then
	echo "Tokenized input dir $INPUT_TOKENIZED_DIR does not exist"
	exit 1
fi

postag_output_dir=$WORKING_DIR/opennlp_tag
# Make sure output dirs exist
mkdir -p $postag_output_dir
mkdir -p $POSTAGS_DIR

postag_temp_dir=$TEMP_DIR/opennlp_tag
# Make sure temp dir exists
mkdir -p $postag_temp_dir

echo "Outputting tags to $postag_output_dir"
echo "Final output in $POSTAGS_DIR"
echo "Documents with an output file in $postag_output_dir will be skipped"
echo

# Parse every file in the input dir
cd $INPUT_TOKENIZED_DIR; dirnames=(*); cd - >/dev/null
for dir in "${dirnames[@]}"; do
	echo ">>>>>>>> FILE: $dir, tagging with OpenNLP <<<<<<<<"
	dir_output_dir=$postag_output_dir/$dir
	mkdir -p $dir_output_dir
	
	final_tar=$POSTAGS_DIR/$dir.tar.gz
	# If the final output file already exists, skip the whole file
	if [ -f $final_tar ]; then
		echo "Output exists, skipping file: $final_tar"
		continue
	fi
	
	echo "Building input file list"
	# Make a list of the files in this directory
	filenames=()
	while IFS= read -d $'\0' -r file ; do
		filenames+=("$file")
	done < <(cd $INPUT_TOKENIZED_DIR/$dir; find * -type f -print0)
	
	# Skip any filenames that already have output
	rm -f $dir_output_dir/to_tag.txt
	if [ -z "$( find $dir_output_dir -type f -name *.txt )" ]; then
		# Output dir is empty: tag everything
		unparsed=("${filenames[@]}")
		printf "%s\n" "${unparsed[@]}" >$dir_output_dir/to_tag.txt
	else
		# Check which files haven't yet been parsed
		echo "  Some files already tagged: checking which (in $dir_output_dir)"
		unparsed=()
		for filename in "${filenames[@]}"; do
			if [ ! -f $dir_output_dir/$filename ]; then
				unparsed=("${unparsed[@]}" "$filename")
				echo "$filename" >>$dir_output_dir/to_tag.txt
			fi
		done
	fi
	
	if [ "${#unparsed[@]}" == "0" ]; then
		echo "No untagged documents"
		parser_exit_status="0"
	else
		echo "Tagging ${#unparsed[@]} documents (of ${#filenames[@]} in input dir), see $dir_output_dir/to_tag.txt"
		
		# Record that we're parsing, in case something goes wrong and we drop out
		touch $dir_output_dir/parsing_started
		
		# Run the parser
		set -o pipefail
		./postag_dir.sh $INPUT_TOKENIZED_DIR/$dir \
						$dir_output_dir \
						$PROCESSES \
						$dir_output_dir/to_tag.txt 2>&1 | tee $dir_output_dir/parse.log
		parser_exit_status="$?"
	fi
	
	# Check the parser completed cleanly
	if [ "$parser_exit_status" != "0" ]; then
		echo "Tagger exited with non-zero status"
		echo "  Refer to $dir_output_dir/parse.log"
	else
		echo "Tagging complete, cleaning up"
		rm -f $dir_output_dir/parsing_started $dir_output_dir/to_tag.txt
		
		# Tar the output and put it in the final dir
		echo "Compressing output and copying to $POSTAGS_DIR"
		( cd $dir_output_dir
		  # Remove the uncompressed output
		  tar -czf $final_tar * && \
		    rm -rf $dir_output_dir )
		echo ">>>>>>>> $dir, finished  <<<<<<<<"
		echo
	fi
done
