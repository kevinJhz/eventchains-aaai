#!/usr/bin/env python
"""
Filter out any files that we've already parsed by looking in the output 
directory.

"""
import argparse
import sys, os
import gzip

parser = argparse.ArgumentParser(description="Filter out filenames " \
			"that correspond to documents to be excluded, listed in " \
			"a file")
parser.add_argument("exclude_file", help="File to get document names " \
			"from (may be Gzipped)")
parser.add_argument("-v", "--invert", action="store_true", 
			help="Invert the filter")
parser.add_argument("-s", "--silent", action="store_true", 
			help="Don't output anything for skipped files")

args = parser.parse_args()
silent = args.silent
invert = args.invert

if args.exclude_file.endswith(".gz"):
	opener = gzip.open
else:
	opener = open

# Load the exclude file
with opener(args.exclude_file, 'r') as exclude_file:
	exclude = frozenset(exclude_file.read().splitlines())

for line in sys.stdin:
	input_filename = line.strip("\n")
	# Get the core of the filename
	document_name = os.path.basename(input_filename).rpartition(".")[0]

	# Check whether this file has been done already
	to_output = document_name not in exclude
	if invert:
		to_output = not to_output
	
	if to_output:
		# This file hasn't been done, so output it to stdout
		print input_filename
	elif not silent:
		# Otherwise, just don't output anything (to stdout) so the 
		#  file gets left out
		print >>sys.stderr, "Excluding %s found in blacklist" % input_filename
