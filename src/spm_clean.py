#! /usr/bin/env python3
# -*- coding: utf-8; -*-
#
# Simple version of the 'spm_decode' command.
#
import sys, os
import re

for line in sys.stdin:
    norm = re.sub(r' ', '', line.strip())
    norm = re.sub(r'\u2581', ' ', norm)
    print(norm.strip())

