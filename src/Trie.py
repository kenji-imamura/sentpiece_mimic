#! /usr/bin/env python3
# -*- coding: utf-8; -*-
#
# Trie module
#
# Copyright (c) 2022 National Institute of Information and Communications Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import sys, os, codecs
import math

###################################
#
# Classes
#

class TrieNode(object):
    def __init__(self):
        self.children = {}
        self.value = None

    def Set(self, seq, freq, lp=-math.inf):
        if len(seq) > 0:
            if seq[0] in self.children:
                next_node = self.children[seq[0]]
                next_node.Set(seq[1:], freq, lp)
        else:
            if self.value is not None:
                self.value['freq'] = freq
                self.value['lp']   = lp

    def Add(self, seq, freq, lp=-math.inf):
        if len(seq) > 0:
            if not seq[0] in self.children:
                self.children[seq[0]] = TrieNode()
            next_node = self.children[seq[0]]
            next_node.Add(seq[1:], freq, lp)
        else:
            if self.value is not None:
                self.value['freq'] += freq
                self.value['lp'] = lp
            else:
                self.value = {'freq': freq, 'lp': lp}

    def Remove(self, seq):
        if len(seq) > 0:
            if seq[0] in self.children:
                next_node = self.children[seq[0]]
                empty_p = next_node.Remove(seq[1:])
                if empty_p:
                    del self.children[seq[0]]
                return (len(self.children) <= 0 and self.value is None)
        else:
            self.value = None
            return len(self.children) <= 0	# empty_p

    def ListAll(self, prefix):
        kv_list = []
        for ch, next_node in self.children.items():
            if next_node.value is not None:
                kv_list.append([prefix + [ch], next_node.value])
            kv_list.extend(next_node.ListAll(prefix + [ch]))
        return kv_list

    def Match(self, seq):
        if len(seq) > 0:
            if seq[0] in self.children:
                next_node = self.children[seq[0]]
                return next_node.Match(seq[1:])
            else:
                return None
        else:
            return self.value

    def Prefix(self, prefix, seq):
        kv_list = []
        if self.value is not None:
            kv_list.append([prefix, self.value])
        if len(seq) > 0:
            if seq[0] in self.children:
                next_node = self.children[seq[0]]
                cnode_list = next_node.Prefix(prefix + [seq[0]], seq[1:])
                kv_list.extend(cnode_list)
        return kv_list


class Trie(object):
    #
    # Root
    #
    def __init__(self):
        self.node = TrieNode()
        self.count = 0

    def __len__(self):
        return self.count

    #
    # Clear
    #
    def Clear(self):
        self.node = TrieNode()
        self.count = 0
    
    #
    # Add one item
    #
    def AddItem(self, word, freq, lp=-math.inf):
        chars = list(word) if isinstance(word, str) else word
        self.node.Add(chars, freq, lp)
        self.count+= 1

    #
    # Set the value to an existing item
    #
    def SetItem(self, word, freq, lp=-math.inf):
        chars = list(word) if isinstance(word, str) else word
        self.node.Set(chars, freq, lp)

    #
    # Remove one item
    #
    def RemoveItem(self, word):
        chars = list(word) if isinstance(word, str) else word
        self.node.Remove(chars)
        self.count -= 1

    #
    # Set probability
    #
    def SetProbability(self):
        kv_list = self.ListAll()
        total_freq = sum([ kv[1]['freq'] for kv in kv_list ])
        count = 0
        for word, value in kv_list:
            rel_freq = value['freq'] / float(total_freq)
            if rel_freq > 0:
                value['lp'] = math.log(rel_freq)
                count += 1
            else:
                self.RemoveItem(word)
        self.count = count
        return count

    #
    # List
    #
    def ListAll(self):
        kv_list = self.node.ListAll([])
        return [ (''.join(kv[0]), kv[1]) for kv in kv_list ]

    #
    # Exact match
    #
    def ExactMatch(self, word):
        chars = list(word) if isinstance(word, str) else word
        return self.node.Match(chars)

    #
    # Prefix search
    #
    def PrefixSearch(self, word):
        chars = list(word) if isinstance(word, str) else word
        kv_list = self.node.Prefix([], chars)
        return [ (''.join(kv[0]), kv[1]) for kv in kv_list ]

    #
    # Save
    #
    def Save(self, file):
        kv_list = self.ListAll()
        with open(file, "w") as ofs:
            for word, value in sorted(kv_list, key=lambda kv: kv[1]['lp'], reverse=True):
                print("{}\t{}\t{}".format(word, value['lp'], value['freq']), file=ofs)

    #
    # Load
    #
    def Load(self, file):
        self.Clear()
        with open(file, "r") as ifs:
            for line in ifs:
                items = line.strip().split("\t")
                if len(items) > 1:
                    lp   = float(items[1])
                    freq = float(items[2]) if len(items) > 2 else 0.0
                    self.AddItem(items[0], freq, lp)

###################################
#
# Main
#

if __name__ == "__main__":
    trie = Trie()
    trie.Load("../tmp/mbart50.vocab")
    print("trie:{}".format(len(trie)))
    xlist = trie.ListAll()
    print("list_all:{}".format(len(xlist)))
    for word in ["テスト", "あかさたな", "ビジネスモデル", "ジネスモデル", "ネスモデル"]:
        print("exact : {} -> {}".format(word, trie.ExactMatch(word)))
        print("prefix: {} -> {}".format(word, trie.PrefixSearch(word)))

