OS := $(shell uname)
ifeq ($(OS), Darwin)
	# Mac
	# -L makes curl follow redirects
	FETCH := curl -L -O
	FETCH_STDOUT := curl -L
else
	# Linux, etc
	FETCH := wget
	FETCH_STDOUT := wget -O -
endif


all :
	$(MAKE) progressbar
	$(MAKE) stemming
	$(MAKE) six.py
	$(MAKE) tabulate.py

progressbar : 
	$(FETCH_STDOUT) https://github.com/WoLpH/python-progressbar/archive/v2.6.2.tar.gz >progressbar.tar.gz
	tar xzf progressbar.tar.gz python-progressbar-2.6.2/progressbar
	mv python-progressbar-2.6.2/progressbar .
	rm -rf progressbar.tar.gz python-progressbar-2.6.2
###################################

arac : 
	$(FETCH) https://github.com/bayerj/arac/archive/master.zip
	unzip master.zip #&& rm master.zip
	cd arac-master && scons

gensim :
	$(FETCH) https://pypi.python.org/packages/source/g/gensim/gensim-0.11.1-1.tar.gz
	tar -zxf gensim-0.11.1-1.tar.gz gensim-0.11.1-1/gensim
	mv gensim-0.11.1-1/gensim . && rm gensim-0.11.1-1.tar.gz

six.py :
	$(FETCH) https://pypi.python.org/packages/source/s/six/six-1.7.3.tar.gz
	tar -xzf six-1.7.3.tar.gz six-1.7.3/six.py
	mv six-1.7.3/six.py . && rm -rf six-1.7.3 six-1.7.3.tar.gz

tabulate.py:
	$(FETCH) https://pypi.python.org/packages/source/t/tabulate/tabulate-0.7.3.tar.gz
	tar -xzf tabulate-0.7.3.tar.gz tabulate-0.7.3/tabulate.py
	mv tabulate-0.7.3/tabulate.py .
	rmdir tabulate-0.7.3

unicodecsv:
	$(FETCH) https://pypi.python.org/packages/source/u/unicodecsv/unicodecsv-0.13.0.tar.gz
	tar -zxf unicodecsv-0.13.0.tar.gz unicodecsv-0.13.0/unicodecsv/
	mv unicodecsv-0.13.0/unicodecsv .
	rm unicodecsv-0.13.0.tar.gz

whim_common:
	git clone git@gitlab.com:markgw/whim-common.git whim_common
