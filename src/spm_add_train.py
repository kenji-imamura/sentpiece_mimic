#! /usr/bin/env python3
# -*- coding: utf-8; -*-
#
# Learn additional subwords and their cost using the EM algorithm.
#
# Copyright (c) 2022 National Institute of Information and Communications Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import sys, os, codecs
import argparse
import collections
import unicodedata
import regex as re
import SentPiece, Trie

#
# Logger
#
import logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("spm_add_train")

#
# Read corpus
#
def read_corpus(file):
    corpus = []
    with open(file, "r") as ifs:
        for line in ifs:
            sent = line.strip()
            corpus.append(sent)
    return corpus

###################################
#
# Main
#

parser= argparse.ArgumentParser(description="Additional training as SentencePiece")

parser.add_argument("-i", "--input" , type=str, dest="input", required=True,
                    help="Normalized input file by spm_normalize (required)")
parser.add_argument("--spm_model" , type=str, dest="spm_model", default="mbart50.vocab",
                    help="Exported SPM model (default:mbart50.vocab)")
parser.add_argument("--add_model" , type=str, dest="model", default="model",
                    help="Additional trained model file (default:model)")
parser.add_argument("-v", "--vocab_size", type=int, dest="vocab_size", default=2000,
                    help="Additional vocabulary size (default: 2000)")
parser.add_argument("--min_freq", type=int, dest="min_freq", default=0,
                    help="Minimum frequency (default: 0")
args = parser.parse_args()

#
# Main
#

spm = SentPiece.SentPiece(base_trie=Trie.Trie())

spm.base_trie.Load(args.spm_model)
logger.info("Basic dict  : {} subwords".format(len(spm.base_trie)))
corpus = read_corpus(args.input)
logger.info("Corpus      : {} lines".format(len(corpus)))

dict_size, atom_size = spm.SetInitialDict(corpus, min_freq=args.min_freq)
logger.info("Initial dict: {} subwords".format(dict_size))
if atom_size > args.vocab_size:
    logger.warning("The vocabulary size ({}) is less than the atom size ({})".format(
        args.vocab_size, atom_size))

spm.SubwordTraining(corpus, args.vocab_size)
spm.trie.Save(args.model)
logger.info("The model has been saved: {}".format(args.model))
