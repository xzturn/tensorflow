# -*- coding: utf-8 -*-

from __future__ import print_function

import codecs
import json
import numpy as np
import os
import re
import time


class CorpusProcessor(object):
    """
    Help to process the Chinese corpus (unicode) for classification
    """

    def __init__(self, encoding='utf-8'):
        self._chs = re.compile(u"[\u4e00-\u9fff]")
        self._encoding = encoding

    def clean(self, text):
        words = re.findall(self._chs, text)
        if words:
            return u" ".join(words)
        return u""

    def _load_data(self, json_file):
        """
        load the corpus (json file), the content is json file like:
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
        print("Processing {} ... ...".format(json_file))
        with codecs.open(json_file, "r", encoding=self._encoding) as fp:
            v = json.load(fp)
        label = v['label']
        data = list()
        for doc in v['documents']:
            data.append(self.clean(doc['content']))
        return label, data

    def load_corpus(self, corpus_dir):
        """
        Load corpus from a directory containing the json files
        """
        x_text, y_label = list(), list()
        tmp = os.listdir(corpus_dir)
        n = len(tmp)
        for t in tmp:
            t_start = time.time()
            json_file = os.path.join(corpus_dir, t)
            i, docs = self._load_data(json_file)
            v = np.zeros(n, dtype=int)
            v[i] = 1
            y_label.extend([list(v) for _ in docs])
            x_text += docs
            t_end = time.time()
            print("\t... ... processing time: {:.3f}s".format(t_end - t_start))
        print("#TEXT: {}, #label: {}".format(len(x_text), len(y_label)))
        return [x_text, np.asarray(y_label)]

    @staticmethod
    def batch_iter(data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]


if __name__ == "__main__":
    from six.moves import cPickle
    import sys

    cp = CorpusProcessor()
    x, y = cp.load_corpus(sys.argv[1])
    with open('x_text.pkl', 'wb') as fp:
        cPickle.dump(x, fp)
    with open('y.pkl', 'wb') as fp:
        cPickle.dump(y, fp)
