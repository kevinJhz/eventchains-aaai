#!../run_py
import argparse
import gzip
import os
from bs4 import BeautifulSoup

parser = argparse.ArgumentParser(description="Read in Gigaword files " \
    "and split into one file per document, containing just the text")
parser.add_argument('files', nargs='+', help='Gigaword files')
parser.add_argument('--list', help="Just list the filenames that " 
        "would be created", action="store_true")
parser.add_argument('--type', help="Restrict the documents to those " \
        "with the given type attribute (e.g. story)")
parser.add_argument('--output', help='Output directory', default="output")
parser.add_argument('--include', help="Only include the files whose "\
        "basename is in the list in this file")

args = parser.parse_args()
just_list = args.list

# Check output dir exists
if not just_list and not os.path.exists(args.output):
        os.makedirs(args.output)
    
# Load the include list
if args.include:
    with open(args.include, 'r') as include_file:
        include = frozenset(include_file.read().splitlines())
else:
    include = None

for filename in args.files:
    # Read in data
    with gzip.open(filename, 'r') as gzip_file:
        xml_data = gzip_file.read()
    #xml_data = xml_data.replace("&AMP;", "&amp;")
    # XML data needs to be wrapped in a top-level tag for minidom
    #xml_data = "<xml>%s</xml>" % xml_data
    
    # Parse the XML data
    #dom = parseString(xml_data)
    soup = BeautifulSoup(xml_data)
    
    for doc_node in soup.find_all("doc"):
        # Check this document is of the required type
        doc_type = doc_node.get("type")
        if args.type is not None and doc_type != args.type:
            continue
        
        # The doc tags have ID attrs that identify the documents
        doc_id = doc_node.get('id')
        filename = "%s.txt" % doc_id
        
        if include:
            # Only include this file if it's in the include list
            if filename not in include:
                continue
        
        if just_list:
            # Just output the filename
            print filename
        else:
            # Create a new file for each document
            doc_filename = os.path.join(args.output, filename)
            with open(doc_filename, 'w') as doc_file:
                # Pull the text out of <P> tags
                for p_node in doc_node.find_all("p"):
                    print >>doc_file, " ".join(p_node.strings)
