#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import collections
import json
import codecs

s = u"这是什么地方，夜是如此的荒凉？"
l = len(s)
counter = collections.Counter(s)
count_pairs = sorted(counter.items(), key=lambda x: -x[1])
print("[{0}] {1}".format(l, s.encode('utf-8')))
print(counter)
print(count_pairs)

chars, _ = zip(*count_pairs)
vocab_size = len(chars)
vocab = dict(zip(chars, range(len(chars))))
print(vocab_size)
print(vocab)

with codecs.open("wechat.35.dat", "r", encoding="utf-8") as fp:
    v = json.load(fp)

data = u""
for doc in v["documents"]:
    data += doc["content"]
    data += u"\n"

print([data])
