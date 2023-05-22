#!/usr/bin/env python
"""
Filter out any files that we've already parsed by looking in the output 
directory.

"""
import argparse
import sys, os

parser = argparse.ArgumentParser(description="Look at an output " \
			"directory to find out what we've done and filter matching " \
			"input files. Filenames are taken from stdin, delimited by " \
			"linebreaks and those not done are output to stdout")
parser.add_argument('output_dir', help="Output directory to look for " \
			".txt output files in")
parser.add_argument('--basenames', help="Just compare to the basenames "\
			"of the files in the output directory", action="store_true")
parser.add_argument('--prefix', help="Prefix path to strip from the " \
			"filenames coming into stdin before matching them to " \
			"paths within the output dir", default="")
parser.add_argument("-a", "--allow-no-output", help="Don't complain and " \
			"exit if the output directory doesn't exist, just output " \
			"all the input files", action="store_true")

args = parser.parse_args()

# Check output dir exists
if not os.path.exists(args.output_dir):
	if args.allow_no_output:
		# No output dir -- just assume nothing's been done
		done_filenames = []
	else:
		print >>sys.stderr, "Output directory %s not found" % args.output_dir
		sys.exit(0)
else:
	# Build a list of filenames in the output directory
	if args.basenames:
		# We can build this list quicker
		done_filenames = sum([
			[filename for filename in filenames if filename.endswith(".txt") ] 
					for (dirpath, dirs, filenames) in os.walk(args.output_dir)], 
			[])
	else:
		done_filenames = sum([
			[os.path.relpath(os.path.join(dirpath, filename), args.output_dir) 
				for filename in filenames if filename.endswith(".txt") ] 
					for (dirpath, dirs, filenames) in os.walk(args.output_dir)], 
			[])
done_filenames = frozenset(done_filenames)

for line in sys.stdin:
	input_filename = line.strip("\n")
	# Remove the prefix path from the filename for matching
	match_filename = input_filename
	if args.prefix:
		match_filename = os.path.relpath(input_filename, args.prefix)

	# Check whether this file has been done already
	if match_filename not in done_filenames:
		# This file hasn't been done, so output it to stdout
		print input_filename
	# Otherwise, just don't output anything so the file gets left out
