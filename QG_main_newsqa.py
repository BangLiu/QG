# -*- coding: utf-8 -*-
"""
Main file for training QANet.
"""
import os
import argparse
import math
import torch
import torch.nn as nn
from datetime import datetime
from data_loader.QG_data import prepro, get_loader
from trainer.QG_trainer import Trainer
from trainer.optim import Optim
from util.file_utils import pickle_load_large_file
from util.exp_utils import set_device, set_random_seed
from util.exp_utils import set_logger, summarize_model


data_folder = "../../../datasets/"

emb_config = {
    "word": {
        # "emb_file": data_folder + "original/Glove/glove.840B.300d.txt",
        # "emb_size": int(2.2e6),
        "emb_file": None,
        "emb_size": 20000,
        "emb_dim": 300,
        "trainable": True,
        "need_conv": False,
        "need_emb": True,
        "is_feature": False},
    "char": {
        "emb_file": None,
        "emb_size": None,
        "emb_dim": 64,
        "trainable": True,
        "need_conv": True,
        "need_emb": True,
        "is_feature": False},
    "bpe": {
        "emb_file": (data_folder +
                     "original/BPE/en.wiki.bpe.op50000.d100.w2v.txt"),
        "emb_size": 50509,
        "emb_dim": 100,
        "trainable": False,
        "need_conv": True,
        "need_emb": True,
        "is_feature": False},
    "pos": {
        "emb_file": None,
        "emb_size": None,
        "emb_dim": 16,
        "trainable": True,
        "need_conv": False,
        "need_emb": True,
        "is_feature": False},
    "ner": {
        "emb_file": None,
        "emb_size": None,
        "emb_dim": 16,
        "trainable": True,
        "need_conv": False,
        "need_emb": True,
        "is_feature": False},
    "iob": {
        "emb_file": None,
        "emb_size": None,
        "emb_dim": 3,
        "trainable": True,
        "need_conv": False,
        "need_emb": True,
        "is_feature": False},
    "dep": {
        "emb_file": None,
        "emb_size": None,
        "emb_dim": 16,
        "trainable": True,
        "need_conv": False,
        "need_emb": True,
        "is_feature": False},
    "answer_iob": {
        "emb_file": None,
        "emb_size": None,
        "emb_dim": 16,
        "trainable": True,
        "need_conv": False,
        "need_emb": True,
        "is_feature": False},
    "is_lower": {
        "emb_file": None,
        "emb_size": None,
        "emb_dim": 16,
        "trainable": True,
        "need_conv": False,
        "need_emb": True,
        "is_feature": True},
    "is_stop": {
        "emb_file": None,
        "emb_size": None,
        "emb_dim": 16,
        "trainable": True,
        "need_conv": False,
        "need_emb": True,
        "is_feature": True},
    "is_punct": {
        "emb_file": None,
        "emb_size": None,
        "emb_dim": 16,
        "trainable": True,
        "need_conv": False,
        "need_emb": True,
        "is_feature": True},
    "is_digit": {
        "emb_file": None,
        "emb_size": None,
        "emb_dim": 16,
        "trainable": True,
        "need_conv": False,
        "need_emb": True,
        "is_feature": True},
    "is_overlap": {
        "emb_file": None,
        "emb_size": None,
        "emb_dim": 16,
        "trainable": True,
        "need_conv": False,
        "need_emb": True,
        "is_feature": True},
    "like_num": {
        "emb_file": None,
        "emb_size": None,
        "emb_dim": 16,
        "trainable": True,
        "need_conv": False,
        "need_emb": True,
        "is_feature": True},
    "is_bracket": {
        "emb_file": None,
        "emb_size": None,
        "emb_dim": 16,
        "trainable": True,
        "need_conv": False,
        "need_emb": True,
        "is_feature": True}}
emb_tags = [
    "word", "answer_iob", "pos", "ner", "dep", "is_lower", "is_digit"]
emb_not_count_tags = {
    "answer_iob": ["B", "I", "O"],
    "is_overlap": [0.0, 1.0]}

parser = argparse.ArgumentParser(description='Lucy')

# experiment
parser.add_argument(
    '--seed', type=int, default=12345)
parser.add_argument(
    '--mode',
    default='train', type=str,
    help='train, eval or test model (default: train)')
parser.add_argument(
    '--visualizer',
    default=False, action='store_true',
    help='use visdom visualizer or not')
parser.add_argument(
    '--log_file',
    default='log.txt',
    type=str, help='path of log file')
parser.add_argument(
    '--no_cuda',
    default=False, action='store_true',
    help='not use cuda')
parser.add_argument(
    '--use_multi_gpu',
    default=False, action='store_true',
    help='use multi-GPU in case there\'s multiple GPUs available')
parser.add_argument(
    '--debug',
    default=False, action='store_true',
    help='debug mode or not')
parser.add_argument(
    '--debug_batchnum',
    default=2, type=int,
    help='only train and test a few batches when debug (default: 2)')

# data
parser.add_argument(
    '--not_processed_data',
    default=False, action='store_true',
    help='whether the dataset already processed')
parser.add_argument(
    '--spacy_not_processed_data',
    default=False, action='store_true',
    help='whether the dataset already processed by spacy')
parser.add_argument(
    '--processed_emb',
    default=False, action='store_true',
    help='whether the embedding files already processed')
parser.add_argument(
    '--train_file',
    default=data_folder + 'original/NewsQA/train.txt',
    type=str, help='path of train dataset')
parser.add_argument(
    '--dev_file',
    default=data_folder + 'original/NewsQA/dev.txt',
    type=str, help='path of dev dataset')
parser.add_argument(
    '--test_file',
    default=data_folder + 'original/NewsQA/test.txt',
    type=str, help='path of test dataset')
parser.add_argument(
    '--train_examples_file',
    default=data_folder + 'processed/NewsQA/train-examples.pkl',
    type=str, help='path of train dataset examples file')
parser.add_argument(
    '--dev_examples_file',
    default=data_folder + 'processed/NewsQA/dev-examples.pkl',
    type=str, help='path of dev dataset examples file')
parser.add_argument(
    '--test_examples_file',
    default=data_folder + 'processed/NewsQA/test-examples.pkl',
    type=str, help='path of test dataset examples file')
parser.add_argument(
    '--train_spacy_processed_examples_file',
    default=data_folder + 'processed/NewsQA/train-spacy-processed-examples.pkl',
    type=str, help='path of train dataset spacy processed examples file')
parser.add_argument(
    '--dev_spacy_processed_examples_file',
    default=data_folder + 'processed/NewsQA/dev-spacy-processed-examples.pkl',
    type=str, help='path of dev dataset spacy processed examples file')
parser.add_argument(
    '--test_spacy_processed_examples_file',
    default=data_folder + 'processed/NewsQA/test-spacy-processed-examples.pkl',
    type=str, help='path of test dataset spacy processed examples file')
parser.add_argument(
    '--train_meta_file',
    default=data_folder + 'processed/NewsQA/train-meta.pkl',
    type=str, help='path of train dataset meta file')
parser.add_argument(
    '--dev_meta_file',
    default=data_folder + 'processed/NewsQA/dev-meta.pkl',
    type=str, help='path of dev dataset meta file')
parser.add_argument(
    '--test_meta_file',
    default=data_folder + 'processed/NewsQA/test-meta.pkl',
    type=str, help='path of test dataset meta file')
parser.add_argument(
    '--train_eval_file',
    default=data_folder + 'processed/NewsQA/train-eval.pkl',
    type=str, help='path of train dataset eval file')
parser.add_argument(
    '--dev_eval_file',
    default=data_folder + 'processed/NewsQA/dev-eval.pkl',
    type=str, help='path of dev dataset eval file')
parser.add_argument(
    '--test_eval_file',
    default=data_folder + 'processed/NewsQA/test-eval.pkl',
    type=str, help='path of test dataset eval file')
parser.add_argument(
    '--train_output_file',
    default=data_folder + 'processed/NewsQA/train_output.txt',
    type=str, help='path of train result file')
parser.add_argument(
    '--eval_output_file',
    default=data_folder + 'processed/NewsQA/eval_output.txt',
    type=str, help='path of evaluation result file')
parser.add_argument(
    '--test_output_file',
    default=data_folder + 'processed/NewsQA/test_output.txt',
    type=str, help='path of test result file')
parser.add_argument(
    '--emb_mats_file',
    default=data_folder + 'processed/NewsQA/emb_mats.pkl',
    type=str, help='path of embedding matrices file')
parser.add_argument(
    '--emb_dicts_file',
    default=data_folder + 'processed/NewsQA/emb_dicts.pkl',
    type=str, help='path of embedding dicts file')
parser.add_argument(
    '--counters_file',
    default=data_folder + 'processed/NewsQA/counters.pkl',
    type=str, help='path of counters file')
parser.add_argument(
    '--glove_word_file',
    default=data_folder + 'original/Glove/glove.840B.300d.txt',
    type=str, help='path of word embedding file')
parser.add_argument(
    '--glove_word_size',
    default=int(2.2e6), type=int,
    help='Corpus size for Glove')
parser.add_argument(
    '--glove_dim',
    default=300, type=int,
    help='word embedding size (default: 300)')
parser.add_argument(
    '--lower',
    default=False, action='store_true',
    help='whether lowercase all texts in data')

# train
parser.add_argument(
    '-b', '--batch_size',
    default=32, type=int,
    help='mini-batch size (default: 32)')
parser.add_argument(
    '-e', '--epochs',
    default=10, type=int,
    help='number of total epochs (default: 20)')
parser.add_argument(
    '--val_num_examples',
    default=10000, type=int,
    help='number of examples for evaluation (default: 10000)')
parser.add_argument(
    '--save_freq',
    default=1, type=int,
    help='training checkpoint frequency (default: 1 epoch)')
parser.add_argument(
    '--save_dir',
    default='checkpoints/', type=str,
    help='directory of saved model (default: checkpoints/)')
parser.add_argument(
    '--resume',
    default='', type=str,
    help='path to latest checkpoint (default: none)')
parser.add_argument(
    '--resume_partial',
    default=False, action='store_true',
    help='whether resume partial pretrained model component(s)')
parser.add_argument(
    '--print_freq',
    default=10, type=int,
    help='print training information frequency (default: 10 steps)')
parser.add_argument(
    '--lr',
    default=0.001, type=float,
    help='learning rate')
parser.add_argument(
    '--lr_warm_up_num',
    default=1000, type=int,
    help='number of warm-up steps of learning rate')
parser.add_argument(
    '--beta1',
    default=0.8, type=float,
    help='beta 1')
parser.add_argument(
    '--beta2',
    default=0.999, type=float,
    help='beta 2')
parser.add_argument(
    '--no_grad_clip',
    default=False, action='store_true',
    help='whether use gradient clip')
parser.add_argument(
    '--max_grad_norm',
    default=5.0, type=float,
    help='global Norm gradient clipping rate')
parser.add_argument(
    '--use_ema',
    default=False, action='store_true',
    help='whether use exponential moving average')
parser.add_argument(
    '--ema_decay',
    default=0.9999, type=float,
    help='exponential moving average decay')
parser.add_argument(
    '--use_early_stop',
    default=True, action='store_false',
    help='whether use early stop')
parser.add_argument(
    '--early_stop',
    default=20, type=int,
    help='checkpoints for early stop')

# model
parser.add_argument(
    '--para_limit',
    default=400, type=int,
    help='maximum context token number')
parser.add_argument(
    '--ques_limit',
    default=50, type=int,
    help='maximum question token number')
parser.add_argument(
    '--ans_limit',
    default=30, type=int,
    help='maximum answer token number')
parser.add_argument(
    '--sent_limit',
    default=100, type=int,
    help='maximum sentence token number')
parser.add_argument(
    '--char_limit',
    default=16, type=int,
    help='maximum char number in a word')
parser.add_argument(
    '--bpe_limit',
    default=6, type=int,
    help='maximum bpe number in a word')
parser.add_argument(
    '--emb_config',
    default=emb_config, type=dict,
    help='config of embeddings')
parser.add_argument(
    '--emb_tags',
    default=emb_tags, type=list,
    help='tags of embeddings that we will use in model')
parser.add_argument(
    '--emb_not_count_tags',
    default=emb_not_count_tags, type=dict,
    help='tags of embeddings that we will not count by counter')

# tmp solution for load issue
parser.add_argument(
    '--d_model',
    default=128, type=int,
    help='model hidden size')
parser.add_argument(
    '--num_head',
    default=8, type=int,
    help='num head')
parser.add_argument(
    '-beam_size', type=int, default=5, help='Beam size')
parser.add_argument(
    '-layers', type=int, default=1,
    help='Number of layers in the LSTM encoder/decoder')
parser.add_argument(
    '-enc_rnn_size', type=int, default=512,
    help='Size of LSTM hidden states')
parser.add_argument(
    '-dec_rnn_size', type=int, default=512,
    help='Size of LSTM hidden states')
parser.add_argument(
    '-att_vec_size', type=int, default=512,
    help='Concat attention vector sizes')
parser.add_argument(
    '-maxout_pool_size', type=int, default=2,
    help='Pooling size for MaxOut layer.')
parser.add_argument(
    '-input_feed', type=int, default=1,
    help="""Feed the context vector at each time step as
    additional input (via concatenation with the word
    embeddings) to the decoder.""")
parser.add_argument(
    '-brnn', action='store_true',
    help='Use a bidirectional encoder')
parser.add_argument(
    '-brnn_merge', default='concat',
    help="""Merge action for the bidirectional hidden states:
    [concat|sum]""")
parser.add_argument(
    '-optim', default='adam',
    help="Optimization method. [sgd|adagrad|adadelta|adam]")
parser.add_argument(
    '-max_weight_value', type=float, default=15,
    help="""If the norm of the gradient vector exceeds this,
    renormalize it to have the norm equal to max_grad_norm""")
parser.add_argument(
    '-dropout', type=float, default=0.5,
    help='Dropout probability; applied between LSTM stacks.')
parser.add_argument(
    '-curriculum', type=int, default=0,
    help="""For this many epochs, order the minibatches based
    on source sequence length. Sometimes setting this to 1 will
    increase convergence speed.""")
parser.add_argument(
    '-extra_shuffle', action="store_true",
    help="""By default only shuffle mini-batch order; when true,
    shuffle and re-assign mini-batches""")
parser.add_argument(
    '-gcn_hidden_size', type=int, default=128,
    help='GCN hidden size.')
parser.add_argument(
    '-gcn_num_layers', type=int, default=2,
    help='GCN number of layers.')
parser.add_argument(
    '--gcn_directed',
    default=False, action='store_true',
    help='whether use directed graph to represent syntax tree')

# learning rate
parser.add_argument(
    '-learning_rate', type=float, default=0.001,
    help="""Starting learning rate. If adagrad/adadelta/adam is
    used, then this is the global learning rate. Recommended
    settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.001""")
parser.add_argument(
    '-learning_rate_decay', type=float, default=0.5,
    help="""If update_learning_rate, decay learning rate by
    this much if (i) perplexity does not decrease on the
    validation set or (ii) epoch has gone past
    start_decay_at""")
parser.add_argument(
    '-start_decay_at', type=int, default=8,
    help="""Start decaying every epoch after and including this
    epoch""")
parser.add_argument('-start_eval_batch', type=int, default=1000,
                    help="""evaluate on dev per x batches.""")
parser.add_argument('-eval_per_batch', type=int, default=500,
                    help="""evaluate on dev per x batches.""")
parser.add_argument(
    '-halve_lr_bad_count', type=int, default=1,
    help="""for change lr.""")

# tricks
parser.add_argument(
    '--tgt_vocab_limit',
    default=20000, type=int,
    help='maximum vocab size of target words in seq2seq')
parser.add_argument(
    '--share_embedder',
    default=False, action='store_true',
    help='encoder decoder share embedder')
parser.add_argument(
    '--use_nonstop_overlap_as_copy_tgt',
    default=False, action='store_true',
    help='whether use non stopword overlap words in input and question as copy tgt')
parser.add_argument(
    '--use_hybrid_clue_tgt',
    default=False, action='store_true',
    help='whether use hybrid strategy as clue prediction tgt')
parser.add_argument(
    '--use_refine_clue',
    default=False, action='store_true',
    help='whether refine predicted clue by word freq')
parser.add_argument(
    '--low_freq_bound',
    default=2000, type=int,
    help='low frequency word bound')
parser.add_argument(
    '--high_freq_bound',
    default=100, type=int,
    help='high frequency word bound')
parser.add_argument(
    '--use_generated_tgt_as_tgt_vocab',
    default=False, action='store_true',
    help='whether use generated tgt as tgt vocab')
parser.add_argument(
    '--use_answer_separate',
    default=False, action='store_true',
    help='whether use answer separate trick')
parser.add_argument(
    '--use_clue_predict',
    default=False, action='store_true',
    help='whether add clue predict module')
parser.add_argument(
    '--clue_predictor',
    default="qanet", type=str,
    help='choose predictor, qanet or gcn based.')
parser.add_argument(
    '--use_refine_copy',
    default=False, action='store_true',
    help='whether refine copy switch and tgt')
parser.add_argument(
    '--use_refine_copy_tgt',
    default=False, action='store_true',
    help='whether refine copy and tgt vocab')
parser.add_argument(
    '--use_refine_copy_src',
    default=False, action='store_true',
    help='whether refine copy and src vocab')
parser.add_argument(
    '--use_refine_copy_tgt_src',
    default=False, action='store_true',
    help='whether refine copy, tgt, and src vocab')
parser.add_argument(
    '--refined_copy_vocab_limit',
    default=2000, type=int,
    help='refined maximum vocab size of copy tgt words in seq2seq')
parser.add_argument(
    '--refined_tgt_vocab_limit',
    default=2000, type=int,
    help='refined maximum vocab size of target words in seq2seq')
parser.add_argument(
    '--refined_src_vocab_limit',
    default=2000, type=int,
    help='refined maximum vocab size of src words in seq2seq')
parser.add_argument(
    '--add_word_freq_emb',
    default=False, action='store_true',
    help='whether add word frequency embedding feature')
parser.add_argument(
    '--word_freq_emb_dim',
    default=32, type=int,
    help='word frequency embedding dimension')
parser.add_argument(
    '--word_freq_threshold',
    default=2000, type=int,
    help='word frequency threshold for low and high frequency')
parser.add_argument(
    '--use_clue_mask',
    default=False, action='store_true',
    help='whether use masking embedding for predicted clue locations')
parser.add_argument(
    '--use_same_lemma_as_overlap',
    default=False, action='store_true',
    help='whether use lemma to get overlap feature')
parser.add_argument(
    '--add_elmo',
    default=False, action='store_true',
    help='whether add elmo embedding')
parser.add_argument(
    '--elmo_options_file', type=str,
    default=data_folder + 'original/ELMo/elmo_2x4096_512_2048cnn_2xhighway_options.json',
    help='path to elmo options_file')
parser.add_argument(
    '--elmo_weight_file', type=str,
    default=data_folder + 'original/ELMo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5',
    help='path to elmo weight_file')
parser.add_argument(
    '--elmo_dropout_prob',
    default=0.1, type=float, help='dropout of elmo')
parser.add_argument(
    '--elmo_requires_grad',
    default=False, action='store_true', help='elmo_requires_grad')
parser.add_argument(
    '--elmo_do_layer_norm',
    default=False, action='store_true', help='do layer norm of elmo')
parser.add_argument(
    '--clue_coef',
    default=1.0, type=float, help='clue loss coef')
parser.add_argument(
    '--data_type',
    default='squad', type=str,
    help='which dataset to use')
parser.add_argument(
    '--net',
    default='s2s_qanet', type=str,
    help='which neural network model to use')


def get_auto_save_dir(args):
    # get checkpoint folder
    checkpoint_folder = []
    checkpoint_folder.append(args.net)
    checkpoint_folder.append(args.data_type)
    if args.use_clue_predict:
        checkpoint_folder.append(args.clue_predictor)
    if args.use_clue_predict:
        checkpoint_folder.append("clue-coef-" + str(args.clue_coef))
    if not args.share_embedder:
        checkpoint_folder.append("not-share-emb")
    if not args.use_ema:
        checkpoint_folder.append("no-ema")
    if args.use_nonstop_overlap_as_copy_tgt:
        checkpoint_folder.append("nonstop-overlap-as-cptgt")
    if args.use_generated_tgt_as_tgt_vocab:
        checkpoint_folder.append("differ-tgtvocab")
    if args.use_hybrid_clue_tgt:
        checkpoint_folder.append(
            "hybrid-clue-" + str(args.high_freq_bound) +
            "-" + str(args.low_freq_bound))
    if args.use_refine_clue:
        checkpoint_folder.append("refine-clue")
    if args.use_refine_copy:
        checkpoint_folder.append(
            "refine-copy-" + str(args.refined_copy_vocab_limit))
    if args.use_refine_copy_tgt:
        checkpoint_folder.append(
            "refine-copy-tgt-" + str(args.refined_copy_vocab_limit) +
            "-" + str(args.refined_tgt_vocab_limit))
    if args.use_refine_copy_tgt_src:
        checkpoint_folder.append(
            "refine-copy-tgt-src-" + str(args.refined_copy_vocab_limit) +
            "-" + str(args.refined_tgt_vocab_limit) +
            "-" + str(args.refined_src_vocab_limit))
    if args.add_word_freq_emb:
        checkpoint_folder.append("wfreq-emb-" + str(args.word_freq_threshold))
    if args.use_clue_mask:
        checkpoint_folder.append("clue-mask")
    if args.add_elmo:
        checkpoint_folder.append("elmo")
    if args.use_same_lemma_as_overlap:
        checkpoint_folder.append("lemma-overlap")
    checkpoint_folder = "_".join(checkpoint_folder)
    checkpoint_folder = args.save_dir + checkpoint_folder + "/"
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    return checkpoint_folder


def main(args):
    if args.net == "s2s_qanet":
        from model.QG_model import QGModel_S2S_CluePredict as Model
    else:
        print("Default use s2s_qanet model.")
        from model.QG_model import QGModel_S2S_CluePredict as Model
    # configuration
    emb_config["word"]["emb_size"] = args.tgt_vocab_limit
    args.emb_config["word"]["emb_size"] = args.tgt_vocab_limit
    args.brnn = True
    args.lower = True

    args.share_embedder = True
    args.use_ema = True
    args.use_clue_predict = True
    args.clue_predictor = "gcn"
    args.use_refine_copy_tgt_src = True
    args.add_word_freq_emb = True

    args.save_dir = get_auto_save_dir(args)
    if args.mode != "train":
        args.resume = args.save_dir + "model_best.pth.tar"  # !!!!! NOTICE: so set --resume won't change it.
    print(args)

    # device, random seed, logger
    device, use_cuda, n_gpu = set_device(args.no_cuda)
    set_random_seed(args.seed)
    logger = set_logger(args.log_file)

    # preprocessing
    if args.not_processed_data:  # use --not_processed_data --spacy_not_processed_data for complete prepro
        prepro(args)

    # data
    emb_mats = pickle_load_large_file(args.emb_mats_file)
    emb_dicts = pickle_load_large_file(args.emb_dicts_file)

    train_dataloader = get_loader(
        args, emb_dicts,
        args.train_examples_file, args.batch_size, shuffle=True)
    dev_dataloader = get_loader(
        args, emb_dicts,
        args.dev_examples_file, args.batch_size, shuffle=False)
    test_dataloader = get_loader(
        args, emb_dicts,
        args.test_examples_file, args.batch_size, shuffle=False)

    # model
    model = Model(args, emb_mats, emb_dicts)
    summarize_model(model)
    if use_cuda and args.use_multi_gpu and n_gpu > 1:
        model = nn.DataParallel(model)
    model.to(device)
    partial_models = None
    partial_resumes = None
    partial_trainables = None

    # optimizer and scheduler
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    for p in parameters:
        if p.dim() == 1:
            p.data.normal_(0, math.sqrt(6 / (1 + p.size(0))))
        elif list(p.shape) == [args.tgt_vocab_limit, 300]:
            print("omit embeddings.")
        else:
            nn.init.xavier_normal_(p, math.sqrt(3))
    optimizer = Optim(
        args.optim, args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        max_weight_value=args.max_weight_value,
        lr_decay=args.learning_rate_decay,
        start_decay_at=args.start_decay_at,
        decay_bad_count=args.halve_lr_bad_count
    )
    optimizer.set_parameters(model.parameters())
    scheduler = None

    loss = {}
    loss["P"] = torch.nn.CrossEntropyLoss()
    loss["D"] = torch.nn.BCEWithLogitsLoss(reduction="sum")

    # trainer
    trainer = Trainer(
        args,
        model,
        train_dataloader=train_dataloader,
        dev_dataloader=dev_dataloader,
        loss=loss,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        emb_dicts=emb_dicts,
        logger=logger,
        partial_models=partial_models,
        partial_resumes=partial_resumes,
        partial_trainables=partial_trainables)

    # start train/eval/test model
    start = datetime.now()
    if args.mode == "train":
        trainer.train()
    elif args.mode == "eval_train":
        args.use_ema = False
        trainer.eval(
            train_dataloader, args.train_eval_file, args.train_output_file)
    elif args.mode in ["eval", "evaluation", "valid", "validation"]:
        args.use_ema = False
        trainer.eval(dev_dataloader, args.dev_eval_file, args.eval_output_file)
    elif args.mode == "test":
        args.use_ema = False
        trainer.eval(
            test_dataloader, args.test_eval_file, args.test_output_file)
    else:
        print("Error: set mode to be train or eval or test.")
    print(("Time of {} model: {}").format(args.mode, datetime.now() - start))


if __name__ == '__main__':
    main(parser.parse_args())
