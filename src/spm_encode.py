#! /usr/bin/env python3
# -*- coding: utf-8; -*-
#
# A clone of 'spm_encode' (after running 'spm_normalize')
#
# Copyright (c) 2022 National Institute of Information and Communications Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import sys, os, codecs
import argparse
import SentPiece

###################################
#
# Main
#

parser= argparse.ArgumentParser(description="Tokenize as SentencePiece.")
parser.add_argument("-m", "--model", type=str, dest="model_file", required=True,
                    help="Model file (required)")
args = parser.parse_args()

#
# Read the model
#
spm = SentPiece.SentPiece()
spm.trie.Load(args.model_file)

#
# Tokenize
#
for line in sys.stdin:
    #sent = spm.NormalizeSimple(line.strip())
    sent = line.strip()
    subwords = spm.Viterbi(sent)
    print(" ".join(subwords))
