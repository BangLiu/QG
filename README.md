# QG
Implementation of the paper "Learning to Generate Questions by Learning What not to Generate"

Run:

> python3 QG_main.py --not_processed_data —spacy_not_processed_data —train

>  python3 QG_main_newsqa.py --not_processed_data —spacy_not_processed_data —train

If debug, add "—debug"

The data only need to be preprocessed for one time. After preprocessing, next time, we need to remove "--not_processed_data —spacy_not_processed_data". In this way, the code will know the data is already preprocessed, and won't do that again.

Please revise the file paths in the main files.

Please download Glove and BPE files from websites.

BPE: <https://github.com/bheinzerling/bpemb#how-to-use-bpemb>

Glove: <https://nlp.stanford.edu/projects/glove/>

SQuAD1.1-Zhou: https://res.qyzhou.me/redistribute.zip

NewsQA:<https://datasets.maluuba.com/NewsQA/dl>

