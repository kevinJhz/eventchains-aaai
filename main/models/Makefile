OS := $(shell uname)
ifeq ($(OS), Darwin)
	# Mac
	FETCH := curl -O
else
	# Linux, etc
	FETCH := wget
endif


all :
	$(MAKE) opennlp
	$(MAKE) wordnet-dict
# 	$(MAKE) TBAQ-cleaned
	$(MAKE) en-sent.bin
	$(MAKE) en-token.bin

########## OPEN NLP ##############
opennlp : 
	mkdir opennlp
	# Get all coref models
	cd opennlp && for filename in acronyms cmodel.bin.gz cmodel.events cmodel.nr.bin.gz defmodel.bin.gz defmodel.events defmodel.nr.bin.gz fmodel.bin.gz fmodel.events fmodel.nr.bin.gz gen.bin.gz gen.events gen.fem gen.mas imodel.bin.gz imodel.events imodel.nr.bin.gz num.bin.gz plmodel.bin.gz plmodel.events plmodel.nr.bin.gz pmodel.bin.gz pmodel.events pmodel.nr.bin.gz pnmodel.bin.gz pnmodel.events pnmodel.nr.bin.gz sim.bin.gz tmodel.bin.gz tmodel.events tmodel.nr.bin.gz; do $(FETCH) http://opennlp.sourceforge.net/models-1.4/english/coref/$$filename; done
	cd opennlp && $(FETCH) http://opennlp.sourceforge.net/models-1.5/en-parser-chunking.bin
	cd opennlp && $(FETCH) http://opennlp.sourceforge.net/models-1.5/en-pos-maxent.bin

en-sent.bin :
	# OpenNLP English sentence detection model
	$(FETCH) http://opennlp.sourceforge.net/models-1.5/en-sent.bin

en-token.bin :
	# OpenNLP English tokenization model
	$(FETCH) http://opennlp.sourceforge.net/models-1.5/en-token.bin
###################################

wordnet-dict :
	$(FETCH) http://wordnetcode.princeton.edu/3.0/WordNet-3.0.tar.gz
	tar -xzf WordNet-3.0.tar.gz WordNet-3.0/dict
	mv WordNet-3.0/dict wordnet-dict
	rm WordNet-3.0.tar.gz

TBAQ-cleaned :
	$(FETCH) http://www.cs.york.ac.uk/semeval-2013/task1/data/uploads/datasets/tbaq-2013-03.zip
	unzip tbaq-2013-03.zip
	rm tbaq-2013-03.zip
