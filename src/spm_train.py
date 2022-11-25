#! /usr/bin/env python3
# -*- coding: utf-8; -*-
#
# Learn subwords and their cost using the EM algorithm.
#
# Copyright (c) 2022 National Institute of Information and Communications Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import sys, os, codecs
import argparse
import SentPiece

#
# Logger
#
import logging
logging.basicConfig(
    format="%(asctime)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("spm_train")

#
# The corpus must be normalized in advance by
# 'spm_normalize --use_internal_normalization --model SPM_MODEL'.
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

parser= argparse.ArgumentParser(description="Training as SentencePiece")

parser.add_argument("-i", "--input" , type=str, dest="input", required=True,
                    help="Normalized input file by spm_normalize (required)")
parser.add_argument("-m", "--model" , type=str, dest="model", default="model",
                    help="Trained model file (default:model)")
parser.add_argument("-v", "--vocab_size", type=int, dest="vocab_size", default=8000,
                    help="Vocabulary size (default: 8000)")
args = parser.parse_args()


#
# Main
#

spm = SentPiece.SentPiece()
corpus = read_corpus(args.input)
logger.info("Corpus      : {} lines".format(len(corpus)))
dict_size, atom_size = spm.SetInitialDict(corpus, min_freq=1)
logger.info("Initial dict: {} subwords".format(dict_size))
if atom_size > args.vocab_size:
    logger.warning("The vocabulary size ({}) is less than the atom size ({})".format(
        args.vocab_size, atom_size))

spm.SubwordTraining(corpus, args.vocab_size)
spm.trie.Save(args.model)
logger.info("The model has been saved: {}".format(args.model))
