# QG
Implementation of the paper "Learning to Generate Questions by Learning What not to Generate"

Run:

> python3 QG_main.py --not_processed_data —spacy_not_processed_data —train

>  python3 QG_main_newsqa.py --not_processed_data —spacy_not_processed_data —train

If debug, add "—debug"

The data only need to be preprocessed for one time. After preprocessing, next time, we need to remove "--not_processed_data —spacy_not_processed_data". In this way, the code will know the data is already preprocessed, and won't do that again.

Please revise the file paths in the main files.

Please download Glove and BPE files from websites.

BPE (maybe updated): <https://github.com/bheinzerling/bpemb#how-to-use-bpemb>

Glove: <https://nlp.stanford.edu/projects/glove/>

Original SQuAD1.1-Zhou: https://res.qyzhou.me/redistribute.zip

Original NewsQA:<https://datasets.maluuba.com/NewsQA/dl>

We put our processed datasets and the package for metric computation in: https://drive.google.com/drive/folders/1csm8AGn0RYD3P6H-5rs0QbXE964x1ono?usp=sharing

You can download the two zip files in the Google Drive link, and unzip them to replace the /Datasets and /nlgeval empty folders in this repository.

