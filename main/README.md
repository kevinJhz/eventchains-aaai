Event chain models codebase
===========================
This codebase contains implementations of the models reported by Granroth-Wilding and Clark (2016, What Happens Next? 
Event Prediction Using a Compositional Neural Network Model, AAAI). It is a slimmed down version of the original 
codebase to cover (roughly) only those things reported in the paper.

Dependencies
------------
Most dependencies are easily installable using the Makefiles in lib/ and lib/python. 

Apart from that, you'll need to install Numpy and Theano for running the models. There are possibly other packages 
you'll need, which would be ideally documented here, but I'm not certain without some investigation.

Event chains pipeline
---------------------
The first step in building any of the models or running the experiments is to run the text processing pipeline to 
extract event chains from a corpus. For the experiments in the paper, following C&J08, this was done on the NYT 
portion of the Gigaword corpus. Since this is a licensed corpus (and since it is large), it is not distributed with 
the code.

To train your models, you'll need to get hold of Gigaword, or use your own text corpus. The event extraction pipeline 
is implemented by the scripts in bin/event_pipeline/. These are probably somewhat difficult to follow, as I opted 
to implement the pipeline as a suite of Bash scripts (it seemed like a good idea at the time).

The basic steps leading up to event extraction are performed by the pipeline script. The ultimate aim is to produce 
what we refer to as "rich docs", which contain event chains for each document, along with entity information and 
the arguments of events (not just their predicates).

Individual stages of the pipeline are run using the pipeline.sh script. The first step is to edit 
bin/event_pipeline/config/local (follow comments there). Then create a pipeline config file: see 
bin/event_pipeline/config/gigaword for an example. Each stage of the pipeline can be called with:

    ./pipeline.sh config/gigaword <stage-number>

Each stage takes a long time to run and is parallelized, according to the config setting. You'll want to run the 
following stages:

    --------------------------------
    | 1   | Plain text extraction  |
    | 2   | Text tokenization      |
    | 3   | Parse with OpenNLP     |
    | 4   | Parse with C&C         |
    | 5   | Coreference resolution |
    --------------------------------

Once these are all done, you've got all the NLP tool output you need, but there's a bunch more processing to be done 
to get the data in a form the models can use. This is not neatly scripted, but the commands I used for Gigaword are 
documented in bin/event_pipeline/rich_docs/gigaword.txt. It shouldn't be too difficult to replicate this (though it 
does take a while to run).

Training models
---------------
Models are trained using the script bin/entity_narrative/train/train_config.sh, together with config files in the 
config/ subdirectory. If you have the dataset prepared as above, it should be reasonably straightforward to run the 
model training by modifying the config files that are there so they use the right dataset directory.

Trained models are stored in the models/ directory.

MCNC evaluation
---------------
The pipeline described above (including the unscripted commands at the end) produces training, dev and test sets. The 
dev and test sets can be used to evaluate trained models on the multiple choice narrative cloze (MCNC) task. 

Before evaluating, you must generate a test sample. For each chain in the test (or dev) corpus, an event is held out 
for the model to predict. Confounders are sampled at random from elsewhere in the corpus. The script 
bin/entity_narrative/eval/experiments/generate_sample.sh, with appropriate modification of paths, can be used to 
do this for the dev set and test set.

That done, you can evaluate a model using the script bin/entity_narrative/eval/experiments/multiple_choice.sh (again, 
you'll need to modify the paths). This is quite quick now, since the sample has already been generated, so it's easy 
to run over and over to compare different models. Note that to avoid accidental foul play with the test set, this 
script by default evaluates on the dev set. To use the test set, use the -t switch. 
