#! /usr/bin/env python

from __future__ import print_function

import datetime
import numpy as np
import os
import sys
import tensorflow as tf
import time

from data_helpers import CorpusProcessor
from six.moves import cPickle
from tensorflow.contrib import learn
from text_cnn import TextCNN


# Parameters
# ==================================================

FLAGS = tf.app.flags.FLAGS

# Data loading params
tf.app.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.app.flags.DEFINE_string("corpus_dir", "corpus", "The corpus directory containing the json files")
tf.app.flags.DEFINE_string("data_dir", "data", "The preprocessed .pkl file generated from the corpus_dir")

# Model Hyperparameters
tf.app.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.app.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.app.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.app.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.app.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.app.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.app.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.app.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.app.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


# Data Preparation
# ==================================================

def _prepare_raw_data():
    # Check directories
    if not os.path.exists(FLAGS.corpus_dir):
        print("CORPUS_DIR: {} not exist!".format(FLAGS.corpus_dir))
        sys.exit(-1)

    if not os.path.exists(FLAGS.data_dir):
        print("DATA_DIR: {} not exist, mkdir for it.".format(FLAGS.data_dir))
        os.mkdir(FLAGS.data_dir)

    x_file = os.path.join(FLAGS.data_dir, "x_text.pkl")
    y_file = os.path.join(FLAGS.data_dir, "y.pkl")

    # Load raw data
    t_start = time.time()
    if os.path.exists(x_file) and os.path.exists(y_file):
        print("Loading preprocessed corpus ... ...")
        with open(x_file, 'rb') as fp:
            x_text = cPickle.load(fp)
        with open(y_file, 'rb') as fp:
            y = cPickle.load(fp)
    else:
        print("Processing corpus ... ...")
        cp = CorpusProcessor()
        x_text, y = cp.load_corpus(FLAGS.corpus_dir)
        with open(x_file, 'wb') as fp:
            cPickle.dump(x_text, fp)
        with open(y_file, 'wb') as fp:
            cPickle.dump(y, fp)
    t_end = time.time()
    print("Load/Process time: {:.3f}s".format(t_end - t_start))
    return x_text, y


def _build_vocabulary(x_text):
    t_start = time.time()
    max_document_length = max([len(x.split(" ")) for x in x_text])
    print("Building vocabulary (with max doc len: {}) ... ...".format(max_document_length))
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    t_end = time.time()
    print("Build vocabulary time: {:.3f}s".format(t_end - t_start))
    return x, vocab_processor


def prepare_data():
    x_file = os.path.join(FLAGS.data_dir, "x.pkl")
    y_file = os.path.join(FLAGS.data_dir, "y.pkl")
    vp_file = os.path.join(FLAGS.data_dir, "vp.bin")
    if os.path.exists(x_file) and os.path.exists(y_file) and os.path.exists(vp_file):
        print("Loading pre-build vocabulary ... ...")
        with open(x_file, 'rb') as fp:
            x = cPickle.load(fp)
        with open(y_file, 'rb') as fp:
            y = cPickle.load(fp)
        vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vp_file)
    else:
        x_text, y = _prepare_raw_data()
        x, vocab_processor = _build_vocabulary(x_text)
        with open(x_file, 'wb') as fp:
            cPickle.dump(x, fp)
        vocab_processor.save(vp_file)
    print("#X: {}, #y: {}".format(len(x), len(y)))

    # Randomly shuffle data
    t_start = time.time()
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    t_end = time.time()
    print("Random shuffle time: {:.3f}s".format(t_end - t_start))

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return vocab_processor, x_train, x_dev, y_train, y_dev


# Training
# ==================================================

def train():
    # prepare data for training
    vocab_processor, x_train, x_dev, y_train, y_dev = prepare_data()

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                      log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.merge_summary(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.scalar_summary("loss", cnn.loss)
            acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_train_batch, y_train_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_train_batch,
                  cnn.input_y: y_train_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_dev_batch, y_dev_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_dev_batch,
                  cnn.input_y: y_dev_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = CorpusProcessor.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


if __name__ == "__main__":
    train()
