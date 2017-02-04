# -*- coding: utf-8 -*-

from __future__ import print_function
import tensorflow as tf

import argparse
import time
import os
from six.moves import cPickle

from utils import HotelCommentLoader
from model import Model


def train(args):
    data_loader = HotelCommentLoader(args.corpus_dir, args.batch_size, args.seq_length)
    data_loader.debug()
    args.vocab_size = data_loader.vocab_size

    # check compatibility if training is continued from previously saved model
    if args.init_from is not None:
        # check if all necessary files exist
        assert os.path.isdir(args.init_from), " %s must be a a path" % args.init_from
        assert os.path.isfile(os.path.join(args.init_from, "config.pkl")),\
            "config.pkl file does not exist in path %s" % args.init_from
        assert os.path.isfile(os.path.join(args.init_from, "chars_vocab.pkl")),\
            "chars_vocab.pkl.pkl file does not exist in path %s" % args.init_from
        ckpt = tf.train.get_checkpoint_state(args.init_from)
        assert ckpt, "No checkpoint found"
        assert ckpt.model_checkpoint_path, "No model path found in checkpoint"

        # open old config and check if models are compatible
        with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
            saved_model_args = cPickle.load(f)
        need_be_same = ["model", "rnn_size", "num_layers", "seq_length"]
        for checkme in need_be_same:
            assert vars(saved_model_args)[checkme] == vars(args)[checkme],\
                "Command line argument and saved model disagree on '%s' " % checkme

        # open saved vocab/dict and check if vocabs/dicts are compatible
        with open(os.path.join(args.init_from, 'chars_vocab.pkl'), 'rb') as f:
            saved_chars, saved_vocab = cPickle.load(f)
        assert saved_chars == data_loader.chars, "Data and loaded model disagree on character set!"
        assert saved_vocab == data_loader.vocab, "Data and loaded model disagree on dictionary mappings!"

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'wb') as f:
        cPickle.dump((data_loader.chars, data_loader.vocab), f)

    def save_model(m_saver, t_sess, save_dir, g_step):
        ckpt_path = os.path.join(save_dir, 'model.ckpt')
        m_saver.save(t_sess, ckpt_path, global_step=g_step)
        print("model saved to {}".format(ckpt_path))

    model = Model(args)

    loss = 0.0
    t_start = time.time()
    with tf.Session() as sess:
        # init variables
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())

        # restore model
        if args.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)
        for e in range(args.num_epochs):
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            data_loader.reset_batch_pointer()
            state = sess.run(model.initial_state)
            for b in range(data_loader.num_batches):
                start = time.time()
                x, y = data_loader.next_batch()
                feed = {model.input_data: x, model.targets: y}
                for i, (c, h) in enumerate(model.initial_state):
                    feed[c] = state[i].c
                    feed[h] = state[i].h
                train_loss, state, _ = sess.run([model.cost, model.final_state, model.train_op], feed)
                end = time.time()
                print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"
                      .format(e * data_loader.num_batches + b,
                              args.num_epochs * data_loader.num_batches,
                              e, train_loss, end - start))
                loss = train_loss
                n_step = e * data_loader.num_batches + b
                if loss <= args.loss_break:
                    save_model(saver, sess, args.save_dir, n_step)
                    args.train_step = n_step
                    break
                elif n_step % args.save_every == 0 \
                        or (e == args.num_epochs - 1 and b == data_loader.num_batches - 1):  # save for the last result
                    save_model(saver, sess, args.save_dir, n_step)
            if loss <= args.loss_break:
                print("train_loss = {} <= {}, training session break!".format(loss, args.loss_break))
                break
    t_end = time.time()
    print("Total train time: {:.3f}".format(t_end - t_start))

    # add some additional info for model training process, and re-store the config.pkl
    args.num_batches = data_loader.num_batches
    args.train_loss = loss
    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_dir', type=str, default='corpus',
                        help='the corpus (json) files directory to learn from')
    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory to store checkpointed models')
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=50,
                        help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000,
                        help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                        help='decay rate for rmsprop')
    parser.add_argument('--loss_break', type=float, default=0.0,
                        help='break down when train loss is less than this value')
    parser.add_argument('--init_from', type=str, default=None,
                        help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                            'config.pkl'        : configuration;
                            'chars_vocab.pkl'   : vocabulary definitions;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    train(parser.parse_args())
