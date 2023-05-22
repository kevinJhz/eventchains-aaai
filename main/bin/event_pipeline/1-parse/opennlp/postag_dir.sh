#!/bin/bash
# Tag a whole directory using OpenNLP
# Directory should contain preprocessed texts, with one sentence per line

if [ $# -lt 3 ]
then
    echo "Usage: postag_dir.sh <input-dir> <output-dir> <processes> <file-list-file>"
    exit 1
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
INPUT_DIR=$( readlink -f $1 )
OUTPUT_DIR=$( readlink -f $2 )
PROCESSES=$3
file_list_file=$4

cd $DIR/../../..
# Split this up into multiple processes
num_files=$( cat $file_list_file | wc -l )
((files_per_process = ($num_files + $PROCESSES - 1) / $PROCESSES ))

split_dir=$OUTPUT_DIR/processes
rm -rf $split_dir
mkdir -p $split_dir

# Split into a batch for each process
split -d -l $files_per_process $file_list_file $split_dir/process-
# Run parallel processes to do the parsing
echo "Running $PROCESSES processes for parsing, up to $files_per_process files each"
for file_list in $split_dir/process-*; do
	# Run a subshell for this process
	(
		let done=0
		while read input_file; do
			# Check that this file's output dir exists
			file_output_dir=$(dirname $OUTPUT_DIR/$input_file)
			if [ ! -d $file_output_dir ]; then mkdir -p $file_output_dir; fi
			
			# Output the outfile instruction
			echo "%% OUTPUT: $OUTPUT_DIR/$input_file"
			# and the input contents itself
			cat $INPUT_DIR/$input_file
			echo -n "." >&2
		done <$file_list | ./run cam.whim.opennlp.PosTag ../models/opennlp/en-pos-maxent.bin
	) &
done

# Wait till they're all done
wait

rm -rf $split_dir
