# -*- coding: utf-8 -*-

from __future__ import print_function

import codecs
import os
import collections
import json
from six.moves import cPickle
import numpy as np


class WechatLoader:
    """
    wechat text loader, the content is json file like:
    {
      "label": int,
      "category": string,
      "documents":
      [
        {"author": string, "title": string, "content": string},
        ...,
        {"author": string, "title": string, "content": string}
      ]
    }
    """

    def __init__(self, corpus_file, batch_size, seq_length, encoding='utf-8'):
        self._data_dir = os.path.join(os.path.abspath("."), "data")
        self._batch_size = batch_size
        self._seq_length = seq_length
        self._encoding = encoding

        if not os.path.exists(self._data_dir):
            os.mkdir(self._data_dir)
        vocab_file = os.path.join(self._data_dir, "vocab.pkl")
        tensor_file = os.path.join(self._data_dir, "data.npy")

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading corpus file")
            self._preprocess(corpus_file, vocab_file, tensor_file)
        else:
            print("loading preprocessed files")
            self._load_preprocessed(vocab_file, tensor_file)
        self._create_batches()
        self._pointer = 0

    def _preprocess(self, corpus_file, vocab_file, tensor_file):
        with codecs.open(corpus_file, "r", encoding=self._encoding) as fp:
            v = json.load(fp)
        data = u""  # MUST be unicode here
        for doc in v["documents"]:
            data = data + doc["title"] + u"\n" + doc["content"] + u"\n\n"
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        self._chars, _ = zip(*count_pairs)
        self._vocab_size = len(self._chars)
        self._vocab = dict(zip(self._chars, range(len(self._chars))))
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self._chars, f)
        self._tensor = np.array(list(map(self._vocab.get, data)))
        np.save(tensor_file, self._tensor)

    def _load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self._chars = cPickle.load(f)
        self._vocab_size = len(self._chars)
        self._vocab = dict(zip(self._chars, range(len(self._chars))))
        self._tensor = np.load(tensor_file)

    def _create_batches(self):
        self._num_batches = int(self._tensor.size / (self._batch_size *
                                                     self._seq_length))

        # When the data (tensor) is too small, let's give them a better error message
        if self._num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        self._tensor = self._tensor[:self._num_batches * self._batch_size * self._seq_length]
        xdata = self._tensor
        ydata = np.copy(self._tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self._x_batches = np.split(xdata.reshape(self._batch_size, -1), self._num_batches, 1)
        self._y_batches = np.split(ydata.reshape(self._batch_size, -1), self._num_batches, 1)

    def reset_batch_pointer(self):
        self._pointer = 0

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def chars(self):
        return self._chars

    @property
    def vocab(self):
        return self._vocab

    @property
    def num_batches(self):
        return self._num_batches

    def next_batch(self):
        x, y = self._x_batches[self._pointer], self._y_batches[self._pointer]
        self._pointer += 1
        return x, y

    def debug(self):
        print("Vocab Size: {0}".format(self._vocab_size))


if __name__ == "__main__":
    import sys
    from optparse import OptionParser

    USAGE = "usage: python utils.py corpus_file -b [batch_size] -s [sequence_length]"
    parser = OptionParser(USAGE)
    parser.add_option("-b", type="int", dest="batch_size", default=50)
    parser.add_option("-s", type="int", dest="seq_length", default=50)
    opt, args = parser.parse_args()

    if len(args) < 1:
        print(USAGE)
        sys.exit(1)

    loader = WechatLoader(args[0], opt.batch_size, opt.seq_length)
    loader.debug()
