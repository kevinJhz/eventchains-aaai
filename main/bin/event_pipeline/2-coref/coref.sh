#!/bin/bash
# Training pipeline stage 4
#
# Call find_all_protagonists.sh with the directories where we put the 
#  parser's output as arguments
#
# Call from pipeline.sh.

shopt -s nullglob

coref_temp_dir=$TEMP_DIR/coref
mkdir -p $coref_temp_dir

echo ">>>>> Document protagonist extraction <<<<<"
echo "Final output dir: $COREF_DIR"
echo "Temp output dir:  $coref_temp_dir"
echo "Pos input dir:    $TAGS_DIR"
echo "Deps input dir:   $DEPS_DIR"
echo "Parse input dir:  $TREES_DIR"

filenames=$( cd $TAGS_DIR && find . -name "*.tar.gz" )
let filenum=0
num_files=$( echo "$filenames" | wc -l )

for findname in $filenames; do
	let filenum=filenum+1
	
	# Remove ./ from pathname
	filename=${findname#./}
	echo
	echo "#### FILE: $filename ($filenum / $num_files) ####"
	base_filename=$( basename $filename .tar.gz )
	file_pos_dir=$coref_temp_dir/pos/$base_filename
	file_dep_dir=$coref_temp_dir/deps/$base_filename
	file_parse_dir=$coref_temp_dir/parse/$base_filename
	
	# A place to put the output
	file_output_dir=$coref_temp_dir/output/$base_filename
	# Where the output all gets archived at the end
	final_tar=$COREF_DIR/$filename
	
	# Check for a file that indicates we've already fully processed this input file
	if [ -f $final_tar ]
	then
		echo "File fully processed previously"
		continue
	fi
	
	# Check that all the necessary input is available
	if [ ! -f $TAGS_DIR/$filename ]; then
		echo "Tag archive not available: $TAGS_DIR/$filename"
		exit 1
	elif [ ! -f $DEPS_DIR/$filename ]; then
		echo "Dependency archive not available: $DEPS_DIR/$filename"
		exit 1
	elif [ ! -f $TREES_DIR/$filename ]; then
		echo "OpenNLP parse archive not available: $TREES_DIR/$filename"
		exit 1
	fi
	
	# Unpack the archives
	mkdir -p $file_pos_dir
	mkdir -p $file_dep_dir
	mkdir -p $file_parse_dir
	tar -xzf $TAGS_DIR/$filename --directory $file_pos_dir
	tar -xzf $DEPS_DIR/$filename --directory $file_dep_dir
	tar -xzf $TREES_DIR/$filename --directory $file_parse_dir
	
	# Process each document extracted from the archive
	# Each document in the pos dir should be mirrored by one in the other dirs
	echo "Building input file list"
	# Make a list of the files in this directory
	docnames=()
	while IFS= read -d $'\0' -r file ; do
		docnames+=("$file")
	done < <(cd $file_pos_dir; find * -type f -name "*.txt" -print0)
	# Don't include to_parse.txt!
	docnames=( ${docnames[@]/to_parse.txt/} )
	
	num_docs=${#docnames[@]}
	if [ $num_docs -eq -0 ]; then
		# No documents in the input -- requires special treatment
		echo "No documents in $TAGS_DIR/$filename"
		echo "Leaving empty output dir: $file_output_dir"
		mkdir -p $file_output_dir
		continue
	fi
	
	# If the (tmp) output dir already exists, maybe it has some of the output already in it
	docnames_to_process=()
	if [ -d $file_output_dir ]; then
		echo "Checking for output in $file_output_dir"
		# Only include input files that don't already have a corresponding output file
		for docnum in "${!docnames[@]}"; do
			docname="${docnames[$docnum]}"
			if [ ! -f $file_output_dir/$docname ]; then
				docnames_to_process[$docnum]=$docname
			fi
		done
	else
		# We need to process all the input files
		docnames_to_process=( "${docnames[@]}" )
	fi
	
	if (( ${#docnames_to_process[@]} ))
	then
		# Process these files in parallel
		echo "Processing ${#docnames_to_process[@]} unprocessed docs (of $num_docs in file), running $PROCESSES in parallel"
		mkdir -p $file_output_dir
		
		# Split up the documents between the processes
		((docs_per_process = (${#docnames_to_process[@]} + $PROCESSES - 1) / $PROCESSES ))
		echo "Up to $docs_per_process documents per process"
		split_dir=$coref_temp_dir/processes
		rm -rf $split_dir
		mkdir -p $split_dir
		# Put the document list for each split in a file
		printf "%s\n" ${docnames_to_process[@]} | split -d -l $docs_per_process - $split_dir/process-
		
		# Set a process going for each batch
		for batch_file in $split_dir/process-*; do
			(
				cd $SCRIPT_DIR
				# Run coreference resolution on all of these documents
				../run cam.whim.coreference.StreamEntitiesExtractor \
					../../models/opennlp \
					$file_pos_dir \
					$file_dep_dir \
					$file_parse_dir \
					$file_output_dir \
					--silent \
					--progress "." \
					<$batch_file
			) &
		done
		# Wait until they're all done
		wait
	fi
		
	# Process output if there was any
	if [ "$( ls -A $file_output_dir/ )" ]; then
		# Check whether we got an output file for every input
		let missing=0
		echo; echo "Checking output in $file_output_dir"
		for docname in ${docnames_to_process[@]}; do 
			if [ ! -f $file_output_dir/$docname ]; then
				let missing=1
				break
			fi
		done
		
		if [ $missing -eq 0 ]; then
			# All outputs are available
			# Don't need to keep the expanded parser output
			rm -rf $file_pos_dir $file_dep_dir $file_parse_dir $split_dir
			
			# Pack the output into a tarball
			echo "Compressing output to $final_tar"
			mkdir -p $COREF_DIR
			(
				cd $file_output_dir
				tar -zcf $final_tar * && \
					rm -rf $file_output_dir
			)
		else
			echo "Not all files were successfully processed: not archiving"
			# Leave coref output for next time...
		fi
	fi
done
