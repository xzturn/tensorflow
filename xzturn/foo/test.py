#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import collections
import jieba

import argparse


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

s1 = [t for t in jieba.cut("这是什么地方，夜是如此的荒凉？北京，北京", cut_all=False)]
s2 = [t for t in jieba.cut("我来自北京清华大学，清华的夜是很漂亮。", cut_all=False)]
s1.extend(s2)
counter = collections.Counter(s1)
count_pairs = sorted(counter.items(), key=lambda x: -x[1])
for t in count_pairs:
    print("{0}: {1}".format(t[0].encode('utf-8'), t[1]))
chars, _ = zip(*count_pairs)
vocab_size = len(chars)
vocab = dict(zip(chars, range(len(chars))))
print(vocab_size)
print(vocab)


parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, default='save',
                    help='model directory to store check-pointed models')
args = parser.parse_args()
args.new = 'abc'
args.v = 1
print(args)
