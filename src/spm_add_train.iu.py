#! /usr/bin/env python3
# -*- coding: utf-8; -*-
#
# Learn additional subwords and their cost using the EM algorithm.
# This file is almost the same as 'spm_add_train.py'
# except for the initialization of the additional dictionary,
# which is only for Inuktutit.
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
# Classes/Functions
#

class SentPieceForInuktutit(SentPiece.SentPiece):
    Inuktutit = re.compile(r'[\p{CanadianAboriginal}]')

    #
    # Initial dictionary only includes Inuktutit words.
    #
    def SetInitialDict(self, corpus, min_freq=0):
        word_freq_hash = collections.defaultdict(float)

        # Count unknown words
        for sent in corpus:
            for i in range(0, len(sent)):
                if self.Inuktutit.match(sent[i:]):
                    if self.base_trie is None or len(self.base_trie.PrefixSearch(sent[i:])) <= 0:
                        self._add_initial_unk(word_freq_hash, sent[i:])

        # Add high-frequent words and compute probabilities
        self.trie.Clear()
        atom_size = 0
        for word, freq in word_freq_hash.items():
            if freq >= min_freq or len(word) == 1:
                self.trie.AddItem(word, freq)
                if len(word) == 1: atom_size += 1
        dict_size = self.trie.SetProbability()
        logger.info("atom_size:{}, dict_size:{}".format(atom_size, dict_size))
        return dict_size, atom_size

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

#spm = SentPiece.SentPiece(base_trie=Trie.Trie())
spm = SentPieceForInuktutit(base_trie=Trie.Trie())

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
