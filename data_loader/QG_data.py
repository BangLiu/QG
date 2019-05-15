# -*- coding: utf-8 -*-
"""
Load SQuAD for QG dataset.
"""
import random
import torch
import codecs
import copy
import numpy as np
from tqdm import tqdm
from collections import Counter
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from .config import *
from util.file_utils import pickle_dump_large_file, pickle_load_large_file
from util.re_utils import get_match_spans
from util.prepro_utils import *
from util.dict_utils import counter2ordered_dict, get_value_maps_between_dicts


def normalize_text(text):
    """
    Notice: don't change the text length.
    Otherwise, the answer position is changed.
    """
    text = text.replace("''", '" ').replace("``", '" ')
    return text


def get_squad_raw_examples(filename, debug=False, debug_length=20):
    """
    Get a list of raw examples given input squad filename.
    """
    print("Start get SQuAD raw examples ...")
    start = datetime.now()
    raw_examples = []
    with codecs.open(filename, encoding="utf-8") as fh:
        lines = fh.readlines()
        num_examples = 0
        for line in tqdm(lines):
            fields = line.strip().split("\t")
            ans_sent = fields[6]
            answer_text = fields[8]
            question = fields[9]

            answer_start_token = int(fields[1].split(" ")[0])
            token_spans = get_token_char_level_spans(
                fields[0], fields[0].split())
            answer_start_in_tokenized_sent = token_spans[answer_start_token][0]

            answer_spans = get_match_spans(answer_text, ans_sent)
            if len(answer_spans) == 0:
                print("pattern: ", answer_text)
                print("match: ", ans_sent)
            answer_start = answer_spans[0][0]
            choice = 0
            gap = abs(answer_start - answer_start_in_tokenized_sent)
            if len(answer_spans) > 1:
                for i in range(len(answer_spans)):
                    new_gap = abs(
                        answer_spans[i][0] - answer_start_in_tokenized_sent)
                    if new_gap < gap:
                        choice = i
                        gap = new_gap
                answer_start = answer_spans[choice][0]

            example = {
                "question": question,
                "ans_sent": ans_sent,
                "answer_text": answer_text,
                "answer_start": answer_start}
            raw_examples.append(example)
            num_examples += 1
            if debug and num_examples >= debug_length:
                break
    print(("Time of get raw examples: {}").format(datetime.now() - start))
    print("Number of raw examples: ", len(raw_examples))
    return raw_examples


def get_newsqa_raw_examples(filename, debug=False, debug_length=20):
    print("Start get NewsQA raw examples ...")
    start = datetime.now()
    raw_examples = []
    with codecs.open(filename, encoding="utf-8") as fh:
        lines = fh.readlines()
        for line in lines:
            fields = line.strip().split("\t")
            ans_sent = fields[2]
            answer_text = fields[4]
            question = fields[3]

            answer_start = int(fields[5].split(":")[0])

            if answer_text not in ans_sent:
                continue

            example = {
                "question": question,
                "ans_sent": ans_sent,
                "answer_text": answer_text,
                "answer_start": answer_start}
            raw_examples.append(example)
            if debug and len(raw_examples) >= debug_length:
                break
    print(("Time of get raw examples: {}").format(datetime.now() - start))
    print("Number of raw examples: ", len(raw_examples))
    return raw_examples


def get_raw_examples(filename, filetype, debug=False, debug_length=20):
    """
    Get a list of raw examples given input filename and file type.
    """
    if filetype.lower() == "squad":
        return get_squad_raw_examples(filename, debug, debug_length)
    elif filetype.lower() == "newsqa":
        return get_newsqa_raw_examples(filename, debug, debug_length)
    else:
        print("Haven't implement loader of " + str(filetype) + " dataset")
        return None


def get_spacy_processed_examples(config, raw_examples,
                                 debug=False, debug_length=20, shuffle=False):
    """
    Get a list of spaCy processed examples given raw examples.
    """
    print("Start transform raw examples to spaCy processed examples...")
    start = datetime.now()
    examples = []
    eval_examples = []
    meta = {}
    meta["num_q"] = 0

    for t in QUESTION_TYPES:
        meta[t] = 0

    for e in tqdm(raw_examples):
        meta["num_q"] += 1

        ans_sent = normalize_text(e["ans_sent"])
        ans_sent_doc = NLP(ans_sent)
        ans_sent_tokens = [token.text for token in ans_sent_doc]
        ans_sent_chars = [list(token) for token in ans_sent_tokens]
        spans = get_token_char_level_spans(ans_sent, ans_sent_tokens)
        ans_sent_syntactic_edges = get_dependency_tree_edges(ans_sent_doc)

        ques = normalize_text(e["question"])
        ques = "<SOS> " + ques + " <EOS>"  # notice: this is important for QG
        ques_doc = NLP(ques)
        ques_tokens = [token.text for token in ques_doc]
        ques_chars = [list(token) for token in ques_tokens]

        answer_text = normalize_text(e["answer_text"])
        answer_start = e["answer_start"]
        answer_end = answer_start + len(answer_text)
        answer_span = []

        for idx, span in enumerate(spans):
            if not (answer_end <= span[0] or
                    answer_start >= span[1]):
                answer_span.append(idx)
        y1_in_sent, y2_in_sent = answer_span[0], answer_span[-1]
        answer_in_sent = " ".join(ans_sent_tokens[y1_in_sent:y2_in_sent + 1])

        example = {
            "question": ques,
            "ques_doc": ques_doc,
            "ques_tokens": ques_tokens,
            "ques_chars": ques_chars,
            "ans_sent": ans_sent,
            "ans_sent_doc": ans_sent_doc,
            "ans_sent_tokens": ans_sent_tokens,
            "ans_sent_chars": ans_sent_chars,
            "ans_sent_syntactic_edges": ans_sent_syntactic_edges,
            "answer": answer_in_sent,
            "y1_in_sent": y1_in_sent,
            "y2_in_sent": y2_in_sent,
            "id": meta["num_q"]}
        examples.append(example)

        if debug and meta["num_q"] >= debug_length:
            break

    if shuffle:
        random.shuffle(examples)

    print(("Time of get spaCy processed examples: {}").format(
        datetime.now() - start))
    print("Number of spaCy processed examples: ", len(examples))
    return examples, meta, eval_examples


def filter_example(config, example, mode="train"):
    """
    Whether filter a given example according to configure.
    :param config: config contains parameters for filtering example
    :param example: an example instance
    :param mode: "train" or "test", they differs in filter restrictions
    :return: boolean
    """
    if mode == "train":
        return (len(example["ans_sent_tokens"]) > config.sent_limit or
                len(example["ques_tokens"]) > config.ques_limit or
                (example["y2_in_sent"] - example["y1_in_sent"]) >
                config.ans_limit)
    elif mode == "test":
        return (len(example["ans_sent_tokens"]) > config.sent_limit or
                len(example["ques_tokens"]) > config.ques_limit)
    else:
        print("mode must be train or test")


def get_filtered_examples(config, examples, mode="train"):
    """
    Get a list of filtered examples according to configure.
    """
    print("Numer of unfiltered examples: ", len(examples))
    filtered_examples = []
    for e in examples:
        if not filter_example(e):
            filtered_examples.append(e)
    print("Numer of filtered examples: ", len(filtered_examples))
    return filtered_examples


def get_lower_word_counter(word_counter):
    specials = ["<PAD>", "<OOV>", "<SOS>", "<EOS>"]
    lower_word_counter = Counter()
    for k, v in word_counter.items():
        if k in specials:
            lower_word_counter[k] += word_counter[k]
        else:
            lower_word_counter[k.lower()] += word_counter[k]
    return lower_word_counter


def get_updated_counters_by_examples(config, counters, examples,
                                     increment=1, init=False, finish=False):
    tags = config.emb_config.keys()
    emb_not_count_tags = config.emb_not_count_tags

    if init:
        counters = init_counters(tags, emb_not_count_tags)
        counters["generated_tgt"] = Counter()

    for e in tqdm(examples):
        update_counters(counters, e["ans_sent_doc"], tags, 1)
        update_counters(counters, e["ques_doc"], tags, 1)
        # update generated_tgt counter here
        stopword_or_non_overlap_ques_tokens = [
            token.text for token in e["ques_doc"] if
            ((token.text not in e["ans_sent_tokens"]) or
             (token.is_stop))]
        for token_text in stopword_or_non_overlap_ques_tokens:
            counters["generated_tgt"][token_text] += increment

    # handle lower case for "word"
    config.lower = True
    if finish and config.lower:
        counters["original_word"] = copy.deepcopy(counters["word"])
        counters["word"] = get_lower_word_counter(counters["word"])
        counters["original_generated_tgt"] = copy.deepcopy(
            counters["generated_tgt"])
        counters["generated_tgt"] = get_lower_word_counter(
            counters["generated_tgt"])
    # TODO: handle numbers, time and so on, don't count in counters
    # bool(re.search(r"^[0-9]+[,.:]*[0-9]*[shm]*$", "1234"))
    return counters


def save(filepath, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
    pickle_dump_large_file(obj, filepath)


def build_linguistic_features(config, example, emb_dicts):
    """
    Given an example, we get its features / tags, and ids.
    """
    fields = ["ques", "ans_sent"]
    fields_cp_token_set = {
        "ques": set(example["ans_sent_tokens"]),
        "ans_sent": set(example["ques_tokens"])}
    length_limits = {
        "ques": config.ques_limit,
        "answer": config.ans_limit,
        "ans_sent": config.sent_limit,
        "word": config.char_limit,
        "bpe": config.bpe_limit}
    tags = config.emb_config.keys()

    for field in fields:
        start, end = example["y1_in_sent"], example["y2_in_sent"]

        for tag in tags:
            field_id = field + "_" + tag + "_ids"
            field_tag = field + "_" + tag
            field_length = len(example[field + "_tokens"])  # unpadded length
            if tag == "word":
                example[field_id] = spacydoc2wids(
                    example[field + "_doc"], emb_dicts[tag],
                    length_limits[field])
            elif tag == "char":
                example[field_id] = spacydoc2cids(
                    example[field + "_doc"], emb_dicts[tag],
                    length_limits[field], length_limits["word"])
            elif tag == "is_overlap":
                example[field_tag] = spacydoc2is_overlap(
                    example[field + "_doc"], fields_cp_token_set[field],
                    length_limits[field], lower=True)
                example[field_id] = feature2ids(
                    example[field_tag], emb_dicts[tag],
                    field_length, length_limits[field])
            elif tag == "bpe":
                example[field_id] = spacydoc2bpeids(
                    example[field + "_doc"], emb_dicts[tag],
                    length_limits[field], length_limits["bpe"])
            elif tag == "answer_iob":
                if field == "ans_sent":
                    example[field_tag] = get_answer_iob(
                        field_length, start, end)
                    example[field_id] = feature2ids(
                        example[field_tag], emb_dicts[tag],
                        field_length, length_limits[field])
            elif tag in ["pos", "ner", "iob", "dep"]:
                example[field_id] = spacydoc2tagids(
                    example[field + "_doc"], tag, emb_dicts[tag],
                    length_limits[field])
            elif tag in ["is_alpha", "is_ascii", "is_digit", "is_lower",
                         "is_title", "is_punct", "is_left_punct",
                         "is_right_punct", "is_bracket", "is_quote",
                         "is_currency", "is_stop", "like_url", "like_num",
                         "like_email"]:
                example[field_tag] = spacydoc2features(
                    example[field + "_doc"], tag, length_limits[field])
                example[field_id] = feature2ids(
                    example[field_tag], emb_dicts[tag],
                    field_length, length_limits[field])
            else:
                print("ERROR: Unsupported tag!")
    # remove <SOS> <EOS> in tgt_tokens
    example["tgt_tokens"] = [x.lower() for x in example["ques_tokens"][1:-1]]
    example["src_tokens"] = [x.lower() for x in example["ans_sent_tokens"]]
    return example


def build_qg_features(config, example, emb_dicts):
    """
    Get example question generation features.
    """
    # emb_dicts["tgt"] embedding index of target tokens in question
    tgt = np.zeros(config.ques_limit, dtype=np.int32)

    # 0/1 indicators of whether each token in question is copied from input
    switch = np.zeros(config.ques_limit, dtype=np.int32)
    # index of copy positions in input for copied question tokens
    copy_position = np.zeros(config.ques_limit, dtype=np.int32)

    switch2 = np.zeros(config.ques_limit, dtype=np.int32)
    copy_position2 = np.zeros(config.ques_limit, dtype=np.int32)

    for idx, tgt_word in enumerate(example["ques_tokens"]):
        tgt_token = example["ques_doc"][idx]
        # get word index
        word_idx = None
        for each in (tgt_word, tgt_word.lower(),
                     tgt_word.capitalize(), tgt_word.upper()):
            if each in emb_dicts["tgt"]:
                word_idx = emb_dicts["tgt"][each]
                break

        # assign values
        if word_idx is None:
            tgt[idx] = emb_dicts["tgt"]["<OOV>"]

            if tgt_word.lower() in example["src_tokens"]:
                switch[idx] = 1
                copy_position[idx] = \
                    example["src_tokens"].index(tgt_word.lower())

            if (tgt_word.lower() in example["src_tokens"] and
                    (not tgt_token.is_stop)):
                switch2[idx] = 1
                copy_position2[idx] = \
                    example["src_tokens"].index(tgt_word.lower())
        else:
            tgt[idx] = word_idx

            if (tgt_word.lower() in example["src_tokens"] and
                    (not tgt_token.is_stop)):
                switch2[idx] = 1
                copy_position2[idx] = \
                    example["src_tokens"].index(tgt_word.lower())
    example["switch"] = switch
    example["copy_position"] = copy_position
    example["switch2"] = switch2
    example["copy_position2"] = copy_position2
    example["tgt"] = tgt

    return example


def get_featured_examples(config, examples, meta, data_type, emb_dicts):
    print("Processing {} examples...".format(data_type))
    total = 0
    total_ = 0
    examples_with_features = []
    for example in tqdm(examples):
        total_ += 1
        if filter_example(config, example, "train"):
            continue
        total += 1

        example = build_linguistic_features(config, example, emb_dicts)
        example = build_qg_features(config, example, emb_dicts)
        examples_with_features.append(example)

    print("Built {} / {} instances of features in total".format(total, total_))
    meta["num_q_filtered"] = total
    return examples_with_features, meta


def init_emb_mat_by_glove(config, emb_mats, emb_dicts):
    with open(config.glove_word_file, "r", encoding="utf-8") as fh:
        vec_size = config.glove_dim
        num_init = 0
        for line in tqdm(fh):
            array = line.split()
            if len(array) < vec_size:
                continue
            word = "".join(array[0:-vec_size])
            vector = list(map(float, array[-vec_size:]))
            if word.lower() in emb_dicts["word"]:  # !!! notice lower here
                idx = emb_dicts["word"][word.lower()]
                emb_mats["word"][idx] = vector
                num_init += 1
        print(num_init, " words initialized by Glove")
    return emb_mats


def prepro(config):
    emb_tags = config.emb_config.keys()
    emb_config = config.emb_config
    emb_mats = {}
    emb_dicts = {}

    debug = config.debug
    debug_length = config.debug_batchnum * config.batch_size

    # get train spacy processed examples
    if config.spacy_not_processed_data:
        train_examples = get_raw_examples(
            config.train_file, config.data_type, debug, debug_length)
        train_examples, train_meta, train_eval = get_spacy_processed_examples(
            config, train_examples, debug, debug_length, shuffle=False)

        dev_examples = get_raw_examples(
            config.dev_file, config.data_type, debug, debug_length)
        dev_examples, dev_meta, dev_eval = get_spacy_processed_examples(
            config, dev_examples, debug, debug_length, shuffle=False)

        test_examples = get_raw_examples(
            config.test_file, config.data_type, debug, debug_length)
        test_examples, test_meta, test_eval = get_spacy_processed_examples(
            config, test_examples, debug, debug_length, shuffle=False)

        save(config.train_spacy_processed_examples_file,
             (train_examples, train_meta, train_eval),
             message="train spacy processed examples and meta")
        save(config.dev_spacy_processed_examples_file,
             (dev_examples, dev_meta, dev_eval),
             message="dev spacy processed examples and meta")
        save(config.test_spacy_processed_examples_file,
             (test_examples, test_meta, test_eval),
             message="test spacy processed examples and meta")
    else:
        train_examples, train_meta, train_eval = pickle_load_large_file(
            config.train_spacy_processed_examples_file)
        dev_examples, dev_meta, dev_eval = pickle_load_large_file(
            config.dev_spacy_processed_examples_file)
        test_examples, test_meta, test_eval = pickle_load_large_file(
            config.test_spacy_processed_examples_file)

    # get counters
    counters = get_updated_counters_by_examples(
        config, None, train_examples, increment=1, init=True, finish=True)
    # only use train data
    final_counters = copy.deepcopy(counters)

    # get emb_mats and emb_dicts
    if not config.processed_emb:
        for tag in emb_tags:
            emb_mats[tag], emb_dicts[tag] = get_embedding(
                final_counters[tag], tag,
                emb_file=emb_config[tag]["emb_file"],
                size=emb_config[tag]["emb_size"],
                vec_size=emb_config[tag]["emb_dim"])
        emb_mats = init_emb_mat_by_glove(config, emb_mats, emb_dicts)
        emb_dicts["tgt"] = emb_dicts["word"]
        emb_mats["generated_tgt"], emb_dicts["generated_tgt"] = get_embedding(
            final_counters["generated_tgt"], "generated_tgt",
            emb_file=emb_config["word"]["emb_file"],
            size=emb_config["word"]["emb_size"],
            vec_size=emb_config["word"]["emb_dim"])
        emb_dicts["generated_tgt2word"] = get_value_maps_between_dicts(
            emb_dicts["generated_tgt"], emb_dicts["word"])
        emb_dicts["word2generated_tgt"] = get_value_maps_between_dicts(
            emb_dicts["word"], emb_dicts["generated_tgt"])
        emb_dicts["idx2tgt"] = {v: k for k, v in emb_dicts["tgt"].items()}
        emb_dicts["idx2generated_tgt"] = {
            v: k for k, v in emb_dicts["generated_tgt"].items()}
    else:
        emb_mats = pickle_load_large_file(config.emb_mats_file)
        emb_dicts = pickle_load_large_file(config.emb_dicts_file)
    for k in emb_dicts:
        print("Embedding dict length: " + k + " " + str(len(emb_dicts[k])))

    # get featured examples
    # TODO: handle potential insert SOS EOS problem
    #       when extracting tag features
    train_examples, train_meta = get_featured_examples(
        config, train_examples, train_meta, "train", emb_dicts)
    dev_examples, dev_meta = get_featured_examples(
        config, dev_examples, dev_meta, "dev", emb_dicts)
    test_examples, test_meta = get_featured_examples(
        config, test_examples, test_meta, "test", emb_dicts)

    # save pickle
    save(config.emb_mats_file, emb_mats, message="embedding mats")
    save(config.emb_dicts_file, emb_dicts, message="embedding dicts")
    save(config.train_examples_file, train_examples, message="train examples")
    save(config.dev_examples_file, dev_examples, message="dev examples")
    save(config.test_examples_file, test_examples, message="test examples")
    save(config.train_meta_file, train_meta, message="train meta")
    save(config.dev_meta_file, dev_meta, message="dev meta")
    save(config.test_meta_file, test_meta, message="test meta")
    save(config.train_eval_file, train_eval, message="train eval")
    save(config.dev_eval_file, dev_eval, message="dev eval")
    save(config.test_eval_file, test_eval, message="test eval")
    save(config.counters_file, final_counters, message="counters")

    # print to txt to debug
    for k in emb_dicts:
        write_dict(emb_dicts[k], "output/emb_dicts_" + str(k) + ".txt")
    for k in counters:
        write_counter(counters[k], "output/counters_" + str(k) + ".txt")
    write_example(train_examples[5], "output/train_example.txt")
    write_example(dev_examples[5], "output/dev_example.txt")
    write_example(test_examples[5], "output/test_example.txt")
    write_dict(train_meta, "output/train_meta.txt")
    write_dict(dev_meta, "output/dev_meta.txt")
    write_dict(test_meta, "output/test_meta.txt")


def write_example(e, filename):
    with codecs.open(filename, mode="w", encoding="utf-8") as fh:
        for k in e:
            if (isinstance(e[k], np.ndarray) or
                    isinstance(e[k], list) or
                    isinstance(e[k], int) or
                    isinstance(e[k], float) or
                    isinstance(e[k], str)):
                fh.write(str(k) + "\n")
                fh.write(str(e[k]) + "\n\n")


def write_dict(d, filename):
    with codecs.open(filename, mode="w", encoding="utf-8") as fh:
        for k in d:
            fh.write(str(k) + " " + str(d[k]) + "\n")


def write_counter(c, filename):
    ordered_c = counter2ordered_dict(c)
    with codecs.open(filename, mode="w", encoding="utf-8") as fh:
        for k in ordered_c:
            fh.write(str(k) + " " + str(ordered_c[k]) + "\n")


def get_word_freq_ids(word_ids, high_freq_bound, low_freq_bound,
                      PAD_id, OOV_id):
    """
    Given numpy array word ids, transform to word freq ids.
    0: PAD, 1: low freq word, 2: mid freq word, 3: high freq word.
    """
    low_freq_ids = (word_ids > low_freq_bound).astype(int) + \
                   (word_ids == OOV_id).astype(int)
    high_freq_ids = (word_ids != PAD_id).astype(int) * \
                    (word_ids != OOV_id).astype(int) * \
                    (word_ids < high_freq_bound).astype(int) * 3
    mid_freq_ids = (word_ids <= low_freq_bound).astype(int) * \
                   (word_ids >= high_freq_bound).astype(int) * 2
    word_freq_ids = low_freq_ids + high_freq_ids + mid_freq_ids
    return word_freq_ids


def get_hybrid_clue_target(example, high_freq_bound, low_freq_bound,
                           PAD_id, OOV_id):
    hybrid_clue_target = (((
        example["ans_sent_is_overlap"] +
        (example["ans_sent_word_ids"] > low_freq_bound).astype(int) +
        (example["ans_sent_word_ids"] == OOV_id).astype(int) +
        (example["ans_sent_ner_ids"] > 4).astype(int) +  # NER_id > 4
        example["ans_sent_is_digit"] +
        example["ans_sent_like_num"]) *
        abs(1 - example["ans_sent_is_stop"]) *
        (example["ans_sent_word_ids"] != PAD_id).astype(int) *
        np.logical_or(
            example["ans_sent_word_ids"] >= high_freq_bound,
            example["ans_sent_word_ids"] == OOV_id).astype(int)
    ) > 0).astype(float)
    return hybrid_clue_target


class QGData(Dataset):

    def __init__(self, config, emb_dicts, examples_file):
        self.examples = pickle_load_large_file(examples_file)
        self.num = len(self.examples)

        # refine examples according to config here.
        start = datetime.now()

        if (config.use_refine_copy or
                config.use_refine_copy_tgt or
                config.use_refine_copy_src or
                config.use_refine_copy_tgt_src):
            assert (config.refined_src_vocab_limit <=
                    config.tgt_vocab_limit)
            assert (config.refined_tgt_vocab_limit <=
                    config.refined_src_vocab_limit)
            assert (config.refined_copy_vocab_limit <=
                    config.refined_tgt_vocab_limit)
            OOV_id = emb_dicts["tgt"]["<OOV>"]
            for i in range(self.num):
                # refine switch and copy_position
                example = self.examples[i]
                switch = np.zeros(config.ques_limit, dtype=np.int32)
                copy_position = np.zeros(config.ques_limit, dtype=np.int32)
                tgt = np.zeros(config.ques_limit, dtype=np.int32)
                # iterate over question tokens
                for idx, tgt_word in enumerate(example["ques_tokens"]):
                    # get question token's word index and generated_tgt index
                    word_idx = None
                    generated_tgt_idx = None
                    for each in (tgt_word, tgt_word.lower(),
                                 tgt_word.capitalize(), tgt_word.upper()):
                        if each in emb_dicts["tgt"]:
                            word_idx = emb_dicts["tgt"][each]
                            generated_tgt_idx = emb_dicts["word2generated_tgt"][word_idx]  
                            break

                    # get refined copy
                    compare_idx = word_idx
                    OOV_idx = emb_dicts["tgt"]["<OOV>"]
                    if config.use_generated_tgt_as_tgt_vocab:
                        compare_idx = generated_tgt_idx
                        OOV_idx = emb_dicts["generated_tgt"]["<OOV>"]

                    # oov or low-freq as copy target
                    if (compare_idx is None) or \
                            (compare_idx >= config.refined_copy_vocab_limit) or \
                            compare_idx == OOV_idx:
                        if tgt_word.lower() in example["src_tokens"]:
                            switch[idx] = 1
                            # NOTICE: we can revise here,
                            # as tgt_word can show multiple times
                            copy_position[idx] = \
                                example["src_tokens"].index(tgt_word.lower())

                    # get refined tgt
                    if (config.use_refine_copy_tgt or
                            config.use_refine_copy_tgt_src):
                        if (compare_idx is None) or \
                                (compare_idx >= config.refined_tgt_vocab_limit) or \
                                compare_idx == OOV_idx:
                            tgt[idx] = OOV_id
                        else:
                            tgt[idx] = word_idx
                # assign new values
                self.examples[i]["switch"] = switch
                self.examples[i]["copy_position"] = copy_position

                # refine tgt ids
                if (config.use_refine_copy_tgt or
                        config.use_refine_copy_tgt_src):
                    self.examples[i]["tgt"] = tgt

                # refine src ids
                if (config.use_refine_copy_src or
                        config.use_refine_copy_tgt_src):
                    c_mask = (example['ans_sent_word_ids'] >=
                              config.refined_src_vocab_limit)
                    self.examples[i]['ans_sent_word_ids'] = \
                        c_mask * OOV_id + \
                        (1 - c_mask) * example['ans_sent_word_ids']
                    q_mask = (example['ques_word_ids'] >=
                              config.refined_src_vocab_limit)
                    self.examples[i]['ques_word_ids'] = \
                        q_mask * OOV_id + \
                        (1 - q_mask) * example['ques_word_ids']

        for i in range(self.num):
            # add elmo embedding
            if config.add_elmo:
                example = self.examples[i]
                self.examples[i]["ans_sent_elmo_ids"] = tokens2ELMOids(
                    example["ans_sent_tokens"], config.sent_limit)
                self.examples[i]["ques_elmo_ids"] = tokens2ELMOids(
                    example["ques_tokens"], config.ques_limit)
            else:
                self.examples[i]["ans_sent_elmo_ids"] = np.array([0])
                self.examples[i]["ques_elmo_ids"] = np.array([0])

            # add word frequency embedding: 0 pad, 1 low-freq, 2 high-freq
            if config.add_word_freq_emb:
                example = self.examples[i]
                self.examples[i]["ans_sent_word_freq"] = get_word_freq_ids(
                    example["ans_sent_word_ids"],
                    config.high_freq_bound, config.low_freq_bound,
                    emb_dicts["word"]["<PAD>"], emb_dicts["word"]["<OOV>"])

                self.examples[i]["ques_word_freq"] = get_word_freq_ids(
                    example["ques_word_ids"],
                    config.high_freq_bound, config.low_freq_bound,
                    emb_dicts["word"]["<PAD>"], emb_dicts["word"]["<OOV>"])
            else:
                self.examples[i]["ans_sent_word_freq"] = np.array([0])
                self.examples[i]["ques_word_freq"] = np.array([0])

            # add hybrid clue target
            if config.use_hybrid_clue_tgt:
                example = self.examples[i]
                self.examples[i]["y_clue"] = get_hybrid_clue_target(
                    example, config.high_freq_bound, config.low_freq_bound,
                    emb_dicts["word"]["<PAD>"], emb_dicts["word"]["<OOV>"])
            else:
                example = self.examples[i]
                self.examples[i]["y_clue"] = \
                    example["ans_sent_is_overlap"] * \
                    abs(1 - example["ans_sent_is_stop"])

        print(("Time of refine data: {}").format(datetime.now() - start))

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        return (self.examples[idx]["ans_sent_word_ids"],
                self.examples[idx]["ans_sent_char_ids"],
                self.examples[idx]["ans_sent_bpe_ids"],
                self.examples[idx]["ans_sent_pos_ids"],
                self.examples[idx]["ans_sent_ner_ids"],
                self.examples[idx]["ans_sent_iob_ids"],
                self.examples[idx]["ans_sent_dep_ids"],
                self.examples[idx]["ans_sent_answer_iob_ids"],
                self.examples[idx]["ans_sent_is_digit"],
                self.examples[idx]["ans_sent_is_digit_ids"],
                self.examples[idx]["ans_sent_is_lower"],
                self.examples[idx]["ans_sent_is_lower_ids"],
                self.examples[idx]["ans_sent_is_punct"],
                self.examples[idx]["ans_sent_is_punct_ids"],
                self.examples[idx]["ans_sent_is_bracket"],
                self.examples[idx]["ans_sent_is_bracket_ids"],
                self.examples[idx]["ans_sent_is_stop"],
                self.examples[idx]["ans_sent_is_stop_ids"],
                self.examples[idx]["ans_sent_like_num"],
                self.examples[idx]["ans_sent_like_num_ids"],
                self.examples[idx]["ans_sent_is_overlap"],
                self.examples[idx]["ans_sent_is_overlap_ids"],
                self.examples[idx]["ans_sent_syntactic_edges"],
                self.examples[idx]["ans_sent_elmo_ids"],
                self.examples[idx]["ans_sent_word_freq"],
                self.examples[idx]["ques_word_ids"],
                self.examples[idx]["ques_char_ids"],
                self.examples[idx]["ques_bpe_ids"],
                self.examples[idx]["ques_pos_ids"],  # !!!!!! notice: we add <SOS> <EOS> to the ques, haven't handle them
                self.examples[idx]["ques_ner_ids"],
                self.examples[idx]["ques_iob_ids"],
                self.examples[idx]["ques_dep_ids"],
                self.examples[idx]["ques_is_digit"],
                self.examples[idx]["ques_is_digit_ids"],
                self.examples[idx]["ques_is_lower"],
                self.examples[idx]["ques_is_lower_ids"],
                self.examples[idx]["ques_is_punct"],
                self.examples[idx]["ques_is_punct_ids"],
                self.examples[idx]["ques_is_bracket"],
                self.examples[idx]["ques_is_bracket_ids"],
                self.examples[idx]["ques_is_stop"],
                self.examples[idx]["ques_is_stop_ids"],
                self.examples[idx]["ques_like_num"],
                self.examples[idx]["ques_like_num_ids"],
                self.examples[idx]["ques_is_overlap"],
                self.examples[idx]["ques_is_overlap_ids"],
                self.examples[idx]["ques_elmo_ids"],
                self.examples[idx]["ques_word_freq"],
                self.examples[idx]["id"],
                self.examples[idx]["y1_in_sent"],
                self.examples[idx]["y2_in_sent"],
                self.examples[idx]["switch"],
                self.examples[idx]["copy_position"],
                self.examples[idx]["switch2"],
                self.examples[idx]["copy_position2"],
                self.examples[idx]["tgt"],
                self.examples[idx]["tgt_tokens"],
                self.examples[idx]["src_tokens"],
                self.examples[idx]["y_clue"])


def collate(data):
    (ans_sent_word_ids, ans_sent_char_ids, ans_sent_bpe_ids,
     ans_sent_pos_ids, ans_sent_ner_ids, ans_sent_iob_ids,
     ans_sent_dep_ids, ans_sent_answer_iob_ids,
     ans_sent_is_digit, ans_sent_is_digit_ids,
     ans_sent_is_lower, ans_sent_is_lower_ids,
     ans_sent_is_punct, ans_sent_is_punct_ids,
     ans_sent_is_bracket, ans_sent_is_bracket_ids,
     ans_sent_is_stop, ans_sent_is_stop_ids,
     ans_sent_like_num, ans_sent_like_num_ids,
     ans_sent_is_overlap, ans_sent_is_overlap_ids,
     ans_sent_syntactic_edges, ans_sent_elmo_ids, ans_sent_word_freq,
     ques_word_ids, ques_char_ids, ques_bpe_ids,
     ques_pos_ids, ques_ner_ids, ques_iob_ids, ques_dep_ids,
     ques_is_digit, ques_is_digit_ids,
     ques_is_lower, ques_is_lower_ids,
     ques_is_punct, ques_is_punct_ids,
     ques_is_bracket, ques_is_bracket_ids,
     ques_is_stop, ques_is_stop_ids,
     ques_like_num, ques_like_num_ids,
     ques_is_overlap, ques_is_overlap_ids,
     ques_elmo_ids, ques_word_freq,
     id, y1_in_sent, y2_in_sent,
     switch, copy_position, switch2, copy_position2,
     tgt, tgt_tokens, src_tokens, y_clue) = zip(*data)
    batch = {}
    batch["ans_sent_word_ids"] = torch.LongTensor(ans_sent_word_ids)
    batch["ans_sent_char_ids"] = torch.LongTensor(ans_sent_char_ids)
    batch["ans_sent_bpe_ids"] = torch.LongTensor(ans_sent_bpe_ids)
    batch["ans_sent_pos_ids"] = torch.LongTensor(ans_sent_pos_ids)
    batch["ans_sent_ner_ids"] = torch.LongTensor(ans_sent_ner_ids)
    batch["ans_sent_iob_ids"] = torch.LongTensor(ans_sent_iob_ids)
    batch["ans_sent_dep_ids"] = torch.LongTensor(ans_sent_dep_ids)
    batch["ans_sent_answer_iob_ids"] = torch.LongTensor(
        ans_sent_answer_iob_ids)
    batch["ans_sent_is_digit"] = torch.FloatTensor(ans_sent_is_digit)
    batch["ans_sent_is_digit_ids"] = torch.LongTensor(ans_sent_is_digit_ids)
    batch["ans_sent_is_lower"] = torch.FloatTensor(ans_sent_is_lower)
    batch["ans_sent_is_lower_ids"] = torch.LongTensor(ans_sent_is_lower_ids)
    batch["ans_sent_is_punct"] = torch.FloatTensor(ans_sent_is_punct)
    batch["ans_sent_is_punct_ids"] = torch.LongTensor(ans_sent_is_punct_ids)
    batch["ans_sent_is_bracket"] = torch.FloatTensor(ans_sent_is_bracket)
    batch["ans_sent_is_bracket_ids"] = torch.LongTensor(
        ans_sent_is_bracket_ids)
    batch["ans_sent_is_stop"] = torch.FloatTensor(ans_sent_is_stop)
    batch["ans_sent_is_stop_ids"] = torch.LongTensor(ans_sent_is_stop_ids)
    batch["ans_sent_like_num"] = torch.FloatTensor(ans_sent_like_num)
    batch["ans_sent_like_num_ids"] = torch.LongTensor(ans_sent_like_num_ids)
    batch["ans_sent_is_overlap"] = torch.FloatTensor(ans_sent_is_overlap)
    batch["ans_sent_is_overlap_ids"] = torch.LongTensor(
        ans_sent_is_overlap_ids)
    batch["ans_sent_syntactic_edges"] = ans_sent_syntactic_edges
    batch["ans_sent_elmo_ids"] = torch.LongTensor(ans_sent_elmo_ids)
    batch["ans_sent_word_freq"] = torch.LongTensor(ans_sent_word_freq)
    batch["ques_word_ids"] = torch.LongTensor(ques_word_ids)
    batch["ques_char_ids"] = torch.LongTensor(ques_char_ids)
    batch["ques_bpe_ids"] = torch.LongTensor(ques_bpe_ids)
    batch["ques_pos_ids"] = torch.LongTensor(ques_pos_ids)
    batch["ques_ner_ids"] = torch.LongTensor(ques_ner_ids)
    batch["ques_iob_ids"] = torch.LongTensor(ques_iob_ids)
    batch["ques_dep_ids"] = torch.LongTensor(ques_dep_ids)
    batch["ques_is_digit"] = torch.FloatTensor(ques_is_digit)
    batch["ques_is_digit_ids"] = torch.LongTensor(ques_is_digit_ids)
    batch["ques_is_lower"] = torch.FloatTensor(ques_is_lower)
    batch["ques_is_lower_ids"] = torch.LongTensor(ques_is_lower_ids)
    batch["ques_is_punct"] = torch.FloatTensor(ques_is_punct)
    batch["ques_is_punct_ids"] = torch.LongTensor(ques_is_punct_ids)
    batch["ques_is_bracket"] = torch.FloatTensor(ques_is_bracket)
    batch["ques_is_bracket_ids"] = torch.LongTensor(ques_is_bracket_ids)
    batch["ques_is_stop"] = torch.FloatTensor(ques_is_stop)
    batch["ques_is_stop_ids"] = torch.LongTensor(ques_is_stop_ids)
    batch["ques_like_num"] = torch.FloatTensor(ques_like_num)
    batch["ques_like_num_ids"] = torch.LongTensor(ques_like_num_ids)
    batch["ques_is_overlap"] = torch.FloatTensor(ques_is_overlap)
    batch["ques_is_overlap_ids"] = torch.LongTensor(ques_is_overlap_ids)
    batch["ques_elmo_ids"] = torch.LongTensor(ques_elmo_ids)
    batch["ques_word_freq"] = torch.LongTensor(ques_word_freq)
    batch["id"] = torch.LongTensor(id)
    batch["y1_in_sent"] = torch.LongTensor(y1_in_sent)
    batch["y2_in_sent"] = torch.LongTensor(y2_in_sent)
    batch["switch"] = torch.FloatTensor(switch)
    batch["copy_position"] = torch.LongTensor(copy_position)
    batch["switch2"] = torch.FloatTensor(switch2)
    batch["copy_position2"] = torch.LongTensor(copy_position2)
    batch["tgt"] = torch.LongTensor(tgt)
    batch["tgt_tokens"] = tgt_tokens
    batch["src_tokens"] = src_tokens
    batch["y_clue"] = torch.FloatTensor(y_clue)
    return batch


def get_loader(config, emb_dicts, examples_file, batch_size, shuffle=True):
    dataset = QGData(config, emb_dicts, examples_file)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate)
    return data_loader
