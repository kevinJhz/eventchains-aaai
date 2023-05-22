OS := $(shell uname)
ifeq ($(OS), Darwin)
	# Mac
	FETCH := curl -L -O
	FETCH_STDOUT := curl -L
else
	# Linux, etc
	FETCH := wget
	FETCH_STDOUT := wget -O -
endif


all :
	$(MAKE) opennlp
	$(MAKE) stanford-parser
	$(MAKE) guava.jar
	$(MAKE) stanford-tagger
	$(MAKE) super-csv-2.1.0.jar
	$(MAKE) argparse4j.jar
	$(MAKE) stemming

########## OPEN NLP ##############
opennlp : opennlp-maxent-3.0.3.jar opennlp-tools-1.5.3.jar opennlp-uima-1.5.3.jar jwnl-1.3.3.jar
	# Clear up downloaded archive
	rm -f apache-opennlp-1.5.3-bin.tar.gz

opennlp-maxent-3.0.3.jar : 
	$(MAKE) apache-opennlp-1.5.3-bin.tar.gz
	tar xzf apache-opennlp-1.5.3-bin.tar.gz apache-opennlp-1.5.3/lib/opennlp-maxent-3.0.3.jar
	mv apache-opennlp-1.5.3/lib/opennlp-maxent-3.0.3.jar .
	rm -rf apache-opennlp-1.5.3
opennlp-tools-1.5.3.jar : 
	$(MAKE) apache-opennlp-1.5.3-bin.tar.gz
	tar xzf apache-opennlp-1.5.3-bin.tar.gz apache-opennlp-1.5.3/lib/opennlp-tools-1.5.3.jar
	mv apache-opennlp-1.5.3/lib/opennlp-tools-1.5.3.jar .
	rm -rf apache-opennlp-1.5.3
opennlp-uima-1.5.3.jar : 
	$(MAKE) apache-opennlp-1.5.3-bin.tar.gz
	tar xzf apache-opennlp-1.5.3-bin.tar.gz apache-opennlp-1.5.3/lib/opennlp-uima-1.5.3.jar
	mv apache-opennlp-1.5.3/lib/opennlp-uima-1.5.3.jar .
	rm -rf apache-opennlp-1.5.3
src/opennlp :
	$(FETCH) http://apache.mirror.anlx.net//opennlp/opennlp-1.5.3/apache-opennlp-1.5.3-src.tar.gz
	tar xzf apache-opennlp-1.5.3-src.tar.gz apache-opennlp-1.5.3-src
	mkdir -p src
	mv apache-opennlp-1.5.3-src src/opennlp
	rm apache-opennlp-1.5.3-src.tar.gz
jwnl-1.3.3.jar : 
	$(MAKE) apache-opennlp-1.5.3-bin.tar.gz
	tar xzf apache-opennlp-1.5.3-bin.tar.gz apache-opennlp-1.5.3/lib/jwnl-1.3.3.jar
	mv apache-opennlp-1.5.3/lib/jwnl-1.3.3.jar .
	rm -rf apache-opennlp-1.5.3

apache-opennlp-1.5.3-bin.tar.gz :
	# Fetch OpenNlp archive
	$(FETCH) http://apache.mesi.com.ar//opennlp/opennlp-1.5.3/apache-opennlp-1.5.3-bin.tar.gz

argparse4j.jar :
	$(FETCH_STDOUT) http://sourceforge.net/projects/argparse4j/files/latest/download?source=files >argparse4j.jar

###################################
	
stanford-parser :
	$(MAKE) stanford-parser-full-2014-01-04.zip
	unzip stanford-parser-full-2014-01-04.zip
	mv stanford-parser-full-2014-01-04/ stanford-parser
	rm stanford-parser-full-2014-01-04.zip

stanford-parser-full-2014-01-04.zip :
	$(FETCH) http://nlp.stanford.edu/software/stanford-parser-full-2014-01-04.zip

stanford-tagger : 
	$(MAKE) stanford-postagger-2014-01-04.zip
	unzip stanford-postagger-2014-01-04.zip
	mv stanford-postagger-2014-01-04 stanford-tagger
	rm stanford-postagger-2014-01-04.zip

stanford-postagger-2014-01-04.zip :
	$(FETCH) http://nlp.stanford.edu/downloads/stanford-postagger-2014-01-04.zip
	
####################################

guava.jar :
	@echo "Fetching Guava"
	$(FETCH_STDOUT) http://search.maven.org/remotecontent?filepath=com/google/guava/guava/15.0/guava-15.0.jar >gauva.jar

###################################
super-csv-2.1.0.jar :
	$(FETCH) http://downloads.sourceforge.net/project/supercsv/2.1.0/super-csv-distribution-2.1.0-bin.zip
	unzip super-csv-distribution-2.1.0-bin.zip super-csv/super-csv-2.1.0.jar
	mv super-csv/super-csv-2.1.0.jar .
	rmdir super-csv
	rm super-csv-distribution-2.1.0-bin.zip

###################################
morph :
	# Fetching morpha
	$(FETCH) http://ilexir.co.uk/media/rasp3os.tar.gz
	tar -zxf rasp3os.tar.gz rasp3os/morph
	mv rasp3os/morph .
	# Building morpha
	cd morph && $(MAKE) morpha

log4j-1.2.17.jar :
	$(FETCH) http://www.us.apache.org/dist/logging/log4j/1.2.17/log4j-1.2.17.tar.gz
	tar -xzf log4j-1.2.17.tar.gz apache-log4j-1.2.17/log4j-1.2.17.jar
	mv apache-log4j-1.2.17/log4j-1.2.17.jar .
	rmdir apache-log4j-1.2.17
	rm log4j-1.2.17.tar.gz

liblinear-java-1.95.jar :
	$(FETCH) http://www.bwaldvogel.de/liblinear-java/liblinear-java-1.95.jar

stemming :
    $(FETCH) https://pypi.python.org/packages/source/s/stemming/stemming-1.0.tar.gz
    tar -xzf stemming-1.0.tar.gz stemming-1.0/stemming/
    mv stemming-1.0/stemming .
    rmdir stemming-1.0
    rm stemming-1.0.tar.gz 
   