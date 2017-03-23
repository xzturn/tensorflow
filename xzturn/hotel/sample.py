# -*- coding: utf-8 -*-

from __future__ import print_function

import tensorflow as tf

import argparse
import os
import random
import time
from six.moves import cPickle
from six import text_type

from model import Model


def sample(args):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
        if args.show_model:
            print(saved_args)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)
    model = Model(saved_args, training=False)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            if len(args.out_file) == 0:
                print(model.sample(sess, chars, vocab, random.randint(args.min, args.max), args.prime, args.sample))
            else:
                with open(args.out_file, 'wb') as f:
                    for i in range(args.n):
                        k = random.randint(args.min, args.max)
                        f.write('\n[%d]\n' % i)
                        ts = time.time()
                        f.write(model.sample(sess, chars, vocab, k, args.prime, args.sample).encode('utf-8'))
                        te = time.time()
                        print('[%d] gen %d len text: %.3fs' % (i, k, te - ts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='save',
                        help='model directory to store check-pointed models')
    parser.add_argument('--show_model', action='store_true', default=False,
                        help='show model arguments before sample')
    parser.add_argument('-n', type=int, default=1,
                        help='number of sample(s) to generate')
    parser.add_argument('--out_file', type=str, default='',
                        help='specify the output file, if empty, output to stdout')
    parser.add_argument('--min', type=int, default=50,
                        help='min number of characters to sample')
    parser.add_argument('--max', type=int, default=200,
                        help='max number of characters to sample')
    parser.add_argument('--prime', type=text_type, default=u' ',
                        help='prime text')
    parser.add_argument('--sample', type=int, default=1,
                        help='0 to use max at each time step, 1 to sample at each time step, 2 to sample on spaces')
    random.seed()
    sample(parser.parse_args())
