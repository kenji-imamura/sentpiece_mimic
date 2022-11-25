#! /usr/bin/env python3
# -*- coding: utf-8; -*-
#
# A clone of SentencePiece
# * Updates probabilities in the dictionary by the EM algorithm.
# * Tokenizes a sentence using Viterbi search.
#
# Copyright (c) 2022 National Institute of Information and Communications Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import sys, os, codecs
import math
import collections
from operator import itemgetter
import unicodedata
import re
from Trie import Trie

UNK_LP = -100.0
UNK_FREQ = math.exp(UNK_LP + 1.0)
EM_LOOP = 2
MAX_LENGTH = 16
WORD_DELIMITERS = ' \u2581,.'
DEBUGGING = False

#
# Logger
#
import logging
logger = logging.getLogger(__name__)

###################################
#
# Functions / Classes
#

#
# LogSumExp
#
def logsumexp(lp1, lp2):
    if lp1 == lp2:
        return lp1 + math.log(2)
    vmax = max(lp1, lp2)
    vmin = min(lp1, lp2)
    if vmax - vmin > 50:
        return vmax
    else:
        return vmax + math.log(math.exp(vmin - vmax) + 1.0)

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

#
# SentencePiece
#
class SentPiece(object):
    #
    # Constructor
    #
    def __init__(self, base_trie=None):
        self.trie = Trie()		# Dynamic dictionary
        self.base_trie = base_trie	# Static dictionary

    #
    # Create an initial dictionary
    #
    def _add_initial_unk(self, hash, sent):
        for j in range(0, len(sent)):
            if j != 0 and sent[j] in WORD_DELIMITERS: break
            if j + 1 <= MAX_LENGTH:
                word = sent[0:j+1]
                hash[word] += 1

    def SetInitialDict(self, corpus, min_freq=0):
        # Count unknown words
        word_freq_hash = collections.defaultdict(float)
        for sent in corpus:
            for i in range(0, len(sent)):
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

    #
    # Match with Trie dictionaies
    #
    def _prefix_search(self, sent):
        kv_list = self.trie.PrefixSearch(sent)
        if self.base_trie is not None:
            kv_list += self.base_trie.PrefixSearch(sent)
        return kv_list

    #
    # Dynamic Programming
    #
    def DPforExpectation(self, sent):
        length = len(sent)
        dp = [ { 'word_lp':[],
                 'fwd_sum':-math.inf, 'bwd_sum':-math.inf }
               for i in range(length+1) ]
        dp[0]['fwd_sum']      = 0.0 
        dp[length]['bwd_sum'] = 0.0 

        # forward processing
        for i in range(length):
            if dp[i]['fwd_sum'] > -math.inf:
                kv_list = self._prefix_search(sent[i:])
                if len(kv_list) > 0:
                    for word, value in kv_list:
                        wlen = len(word)
                        lp = dp[i]['fwd_sum'] + value['lp']
                        dp[i+wlen]['fwd_sum'] = logsumexp(dp[i+wlen]['fwd_sum'], lp)
                        dp[i+wlen]['word_lp'].append((word, value, lp))
	        # UNK
                else:
                    for wlen in range(1, min(MAX_LENGTH+1, length-i+1)):
                        lp = dp[i]['fwd_sum'] + UNK_LP
                        dp[i+wlen]['fwd_sum'] = logsumexp(dp[i+wlen]['fwd_sum'], lp)
                        if i+wlen >= length or len(self._prefix_search(sent[i+wlen:])) > 0: break

        # backward processing
        for i in range(length, 0, -1):
            if dp[i]['bwd_sum'] > -math.inf:
                kvp_list = dp[i]['word_lp']
                if len(kvp_list) > 0:
                    for word, value, _ in kvp_list:
                        wlen = len(word)
                        lp = dp[i]['bwd_sum'] + value['lp']
                        dp[i-wlen]['bwd_sum'] = logsumexp(dp[i-wlen]['bwd_sum'], lp)
                # UNK
                else:
                    for wlen in range(1, min(MAX_LENGTH+1, i+1)):
                        lp = dp[i]['bwd_sum'] + UNK_LP
                        dp[i-wlen]['bwd_sum'] = logsumexp(dp[i-wlen]['bwd_sum'], lp)
                        if i-wlen <= 0 or len(self._prefix_search(sent[i-wlen-1:])) > 0: break
        return dp

    def DPforViterbi(self, sent):
        length = len(sent)
        dp = [ { 'max_lp' :-math.inf, 'max_pos':-1 }
               for i in range(length+1) ]
        dp[0]['max_lp']       = 0.0 

        # forward processing
        for i in range(length):
            kv_list = self._prefix_search(sent[i:])
            if len(kv_list) > 0:
                for word, value in kv_list:
                    wlen = len(word)
                    max_lp = dp[i]['max_lp'] + value['lp']
                    if max_lp > dp[i+wlen]['max_lp']:
                        dp[i+wlen]['max_lp'] = max_lp
                        dp[i+wlen]['max_pos'] = i
	    # UNK
            else:
                for wlen in range(1, min(MAX_LENGTH+1, length-i+1)):
                    max_lp = dp[i]['max_lp'] + UNK_LP
                    if max_lp > dp[i+wlen]['max_lp']:
                        dp[i+wlen]['max_lp'] = max_lp
                        dp[i+wlen]['max_pos'] = i
                    if len(self._prefix_search(sent[i+wlen:])) > 0: break
        return dp

    #
    # Expectation
    #
    def Expectation(self, sent):
        length = len(sent)
        dp = self.DPforExpectation(sent)
        sent_lp = dp[length]['fwd_sum']
        if DEBUGGING:
            logger.info("*** sent_loss: {} -- db: {}".format(-sent_lp, dp))
            for i in range(0, length + 1):
                logger.info("[{}] fwd: {}, bwd:{}".format(i, dp[i]['fwd_sum'], dp[i]['bwd_sum'], ))

        word_freq_hash = collections.defaultdict(float)
        for i in range(1, length+1):
            summed_lp = dp[i]['fwd_sum']
            for word, value, lp in dp[i]['word_lp']:
                loss = dp[i - len(word)]['fwd_sum'] + dp[i]['bwd_sum'] + value['lp'] - sent_lp
                word_freq_hash[word] += math.exp(loss)
                if DEBUGGING:
                    logger.info("word: {}, loss: {}, freq:{}".format(word, loss, word_freq_hash[word]))

        if DEBUGGING: logger.info("*** word_freq_hash: {}".format(word_freq_hash))
        return -sent_lp, word_freq_hash

    #
    # Viterbi search
    #
    def Viterbi(self, sent):
        dp = self.DPforViterbi(sent)
        subwords = []
        i = len(sent)
        while i > 0:
            pos = dp[i]['max_pos']
            subwords.append(sent[pos:i])
            i = pos
        subwords.reverse()
        return subwords

    #
    # Estimate word frequencies
    #
    def EstimatedWordFreq(self, corpus):
        word_freq_hash = collections.defaultdict(float)
        current_loss = 0.0
        for sent in corpus:
            loss, wf_hash = self.Expectation(sent)
            current_loss += loss
            for w, f in wf_hash.items():
                word_freq_hash[w] += f

        # Remain only words that are included in the 'trie' dictionary.
        word_keys = list(word_freq_hash.keys())
        for word in word_keys:
            if self.trie.ExactMatch(word) is None:
                del word_freq_hash[word]

        return word_freq_hash, current_loss

    #
    # Optimization by EM algorithm
    # This method only change the 'trie' dictionary.
    #
    def EmOptimization(self, corpus):
        prev_loss = None
        for iter in range(EM_LOOP):
            word_freq_hash, current_loss = self.EstimatedWordFreq(corpus)
            for word, freq in word_freq_hash.items():
                if freq <= 0:
                    if len(word) > 1:
                        self.trie.RemoveItem(word)
                    else:
                        self.trie.SetItem(word, UNK_FREQ)
                else:
                    self.trie.SetItem(word, max(freq, UNK_FREQ))
            self.trie.SetProbability()

            logger.info("[{}] loss: {}".format(iter+1, current_loss))
            if prev_loss is not None and prev_loss - current_loss < 0.00001:
                break
            prev_loss = current_loss

    #
    # Training
    #
    def SubwordTraining(self, corpus, num_subwords=4000):
        for iter in range(1000):
            logger.info("*** Iteration {}".format(iter))
            self.EmOptimization(corpus)
            word_value_list = self.trie.ListAll()
            if len(word_value_list) <= num_subwords: break

            atom_wv_list  = [ wv for wv in word_value_list if len(wv[0]) == 1 ]
            multi_wv_list = sorted([ wv for wv in word_value_list if len(wv[0]) > 1 ],
                                   key=lambda kv: kv[1]['freq'], reverse=True)
            atom_len  = len(atom_wv_list)
            multi_len = len(multi_wv_list)

            new_total_len = int(len(word_value_list) * 0.8)
            if new_total_len < num_subwords:
                new_total_len = num_subwords
            new_multi_len = new_total_len - atom_len
            new_multi_len = min(len(multi_wv_list), max(0, new_multi_len))

            trie = Trie()
            for word, value in atom_wv_list:
                trie.AddItem(word, max(value['freq'], UNK_FREQ))
            if new_multi_len > 0:
                for word, value in multi_wv_list[0:new_multi_len]:
                    trie.AddItem(word, max(value['freq'], UNK_FREQ))
            trie.SetProbability()
            self.trie = trie
            if new_multi_len <= 0: break
            logger.info("current_subwords: {}".format(new_total_len))

    #
    # Sentence normalizer for debugging
    # This is not identical to the normalizer of the original SentencePiece.
    #
    def NormalizeSimple(self, line):
        xline = unicodedata.normalize('NFKC', line)
        xline = re.sub(r'[\u200b\200d\ufffd\ufeff]', ' ', xline)
        xline = re.sub(r'\s+', ' ', xline)
        xline = xline.replace(' ', '\u2581')
        return xline.strip()

###################################
#
# Main
#

if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )
    logger = logging.getLogger("SentPiece")
    

    spm = SentPiece()
    corpus = read_corpus("../test/sub.norm.ja")
    count, _ = spm.SetInitialDict(corpus, min_freq=1)
    print("*** Initialized: {}".format(count))

    def debug_expectation(sent):
        global DEBUGGING
        DEBUGGING = True
        ret = spm.Expectation(sent)
        DEBUGGING = False
        return ret

    debug_expectation(spm.NormalizeSimple(" ビジネスモデル"))
#    debug_expectation(spm.NormalizeSimple(" business model"))

    spm.SubwordTraining(corpus, 8000)
    debug_expectation(spm.NormalizeSimple(" ビジネスモデル"))
#    debug_expectation(spm.NormalizeSimple(" business model"))

    print()
    for sent in ['ビジネス', '経済', 'ビジネスモデル',
                 '現在,筋ジストロフィー患者の移動介助において文書マニュアルを使用している。']:
#    for sent in ['At present, the document manual is used in transfer assistance of the muscular dystrophy patient.']:
        nsent = spm.NormalizeSimple(sent)
        subwords = spm.Viterbi(nsent)
        print("viterbi: {} -- {}".format(nsent, " ".join(subwords)))
            
    spm.trie.Save("trie.sub.ja")
    for sent in corpus:
        subwords = spm.Viterbi(sent)
        print("{}\t{}".format(sent, subwords))
