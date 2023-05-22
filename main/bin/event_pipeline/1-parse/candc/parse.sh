#!/bin/bash
# Parse the whole input data, outputting parse trees.
# This version uses the C&C parser.
# Call from pipeline.sh.

# Check input dir exists
if [ ! -d "$INPUT_TOKENIZED_DIR" ]
then
	echo "Tokenized input dir $INPUT_TOKENIZED_DIR does not exist"
	exit 1
fi

parse_output_dir=$WORKING_DIR/candc
# Make sure output dirs exist
mkdir -p $parse_output_dir
mkdir -p $DEPS_DIR
mkdir -p $TAGS_DIR

deps_output_dir=$parse_output_dir/deps
mkdir -p $deps_output_dir
tags_output_dir=$parse_output_dir/tags
mkdir -p $tags_output_dir

echo "Outputting trees to $deps_output_dir and $tags_output_dir"
echo "Final output in $DEPS_DIR and $TAGS_DIR"
echo "Documents with an output file in $parse_output_dir will be skipped"
echo

# Parse every file in the input dir
cd $INPUT_TOKENIZED_DIR; dirnames=(*); cd - >/dev/null
for dir in "${dirnames[@]}"; do
	echo ">>>>>>>> FILE: $dir, parsing with C&C <<<<<<<<"
	dir_deps_output_dir=$deps_output_dir/$dir
	dir_tags_output_dir=$tags_output_dir/$dir
	mkdir -p $dir_deps_output_dir
	mkdir -p $dir_tags_output_dir
	
	final_deps_tar=$DEPS_DIR/$dir.tar.gz
	final_tags_tar=$TAGS_DIR/$dir.tar.gz
	# If the final output file already exists, skip the whole file
	if [ -f $final_deps_tar ] && [ -f $final_tags_tar ]; then
		echo "Output exists, skipping file: $final_deps_tar and $final_tags_tar"
		continue
	fi
	
	echo "Building input file list"
	# Make a list of the files in this directory
	filenames=()
	while IFS= read -d $'\0' -r file ; do
		filenames+=("$file")
	done < <(cd $INPUT_TOKENIZED_DIR/$dir; find * -type f -print0)
	
	# Skip any filenames that already have output
	rm -f $dir_tags_output_dir/to_parse.txt
	if [ -z "$(find $dir_deps_output_dir -type f -name *.txt)" ] || \
			[ -z "$(find $dir_tags_output_dir -type f -name *.txt)" ]; then
		# One of output dirs is empty: parse everything
		echo "Parsing all input files"
		unparsed=("${filenames[@]}")
		printf "%s\n" "${unparsed[@]}" >$dir_tags_output_dir/to_parse.txt
	else
		# Check which files haven't yet been parsed
		echo "  Some files already parsed: checking which (in $dir_deps_output_dir and $dir_tags_output_dir)"
		unparsed=()
		for filename in "${filenames[@]}"; do
			if [ ! -f $dir_deps_output_dir/$filename ] || \
					[ ! -f $dir_tags_output_dir/$filename ]; then
				# If either of the output files isn't available, parse this
				unparsed=("${unparsed[@]}" "$filename")
				echo "$filename" >>$dir_tags_output_dir/to_parse.txt
			fi
		done
	fi
	
	if [ "${#unparsed[@]}" == "0" ]; then
		echo "No unparsed documents"
		parser_exit_status="0"
	else
		echo "Parsing ${#unparsed[@]} documents (of ${#filenames[@]} in input dir), see $dir_tags_output_dir/to_parse.txt"
		
		# Record that we're parsing, in case something goes wrong and we drop out
		touch $dir_tags_output_dir/parsing_started
		
		# Run the parser
		# Give the pool a name so we can run multiple instances at once
		./parse_dir.sh --dep-dir $dir_deps_output_dir \
						--pos-dir $dir_tags_output_dir \
						--processes $PROCESSES \
						--include $dir_tags_output_dir/to_parse.txt \
						--log $dir_tags_output_dir/parse.log \
						--progress \
						--pool-name $PIPELINE_NAME \
						$INPUT_TOKENIZED_DIR/$dir $EXTRA_CANDC_ARGS
		parser_exit_status="$?"
	fi
	
	# Check the parser completed cleanly
	if [ "$parser_exit_status" != "0" ]; then
		echo "Parser exited with non-zero status"
		echo "  Refer to $dir_tags_output_dir/parse.log"
	else
		echo "Parsing complete, cleaning up"
		# Overwrite the list of files to parse so that if we rerun we know that the whole file's 
		#  been done, even if we're not recomputing the file list
		echo -n >$dir_tags_output_dir/to_parse.txt
		rm -f $dir_tags_output_dir/parsing_started
		
		# Tar the output and put it in the final dir
		echo "Compressing output and copying to $DEPS_DIR and $TAGS_DIR"
		# Tags output
		( cd $dir_tags_output_dir
		  # Compress and remove the uncompressed output
		  tar -czf $final_tags_tar * && \
		    rm -rf $dir_tags_output_dir ) &
		# Deps output
		( cd $dir_deps_output_dir
		  tar -czf $final_deps_tar * && \
		    rm -rf $dir_deps_output_dir ) &
		wait
		echo ">>>>>>>> $dir, finished  <<<<<<<<"
		echo
	fi
done
