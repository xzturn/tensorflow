#! /usr/bin/env python

from __future__ import print_function

import csv
import numpy as np
import os
import sys
import tensorflow as tf
import time

from data_helpers import CorpusProcessor
from tensorflow.contrib import learn


# Parameters
# ==================================================

FLAGS = tf.app.flags.FLAGS

# Data Parameters
tf.app.flags.DEFINE_string("corpus_dir", "corpus", "The corpus directory containing the json files")

# Eval Parameters
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.app.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.app.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.app.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


# Data Preparation
# ==================================================

def prepare_data():
    # Load eval data, maybe we evaluate on all training data
    if not os.path.exists(FLAGS.corpus_dir):
        print("CORPUS_DIR: {} not exist!".format(FLAGS.corpus_dir))
        sys.exit(-1)

    print("Loading evaluation data ... ...")
    t_start = time.time()
    cp = CorpusProcessor()
    x_raw, y_test = cp.load_corpus(FLAGS.corpus_dir)
    y_test = np.argmax(y_test, axis=1)
    t_end = time.time()
    print("Load time: {:.3f}s".format(t_end - t_start))

    # Map data into vocabulary
    print("Mapping data into vocabulary ... ...")
    t_start = time.time()
    vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_raw)))
    t_end = time.time()
    print("Map time: {:.3f}s".format(t_end - t_start))

    return x_raw, x_test, y_test


# Evaluation
# ==================================================

def evaluate():
    x_raw, x_test, y_test = prepare_data()

    print("\nEvaluating...\n")
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                      log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = CorpusProcessor.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []

            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

    # Print accuracy if y_test is defined
    if y_test is not None:
        correct_predictions = float(sum(all_predictions == y_test))
        print("Total number of test examples: {}".format(len(y_test)))
        print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

    # Save the evaluation to a csv
    predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
    out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
    print("Saving evaluation to {0}".format(out_path))
    with open(out_path, 'w') as f:
        csv.writer(f).writerows(predictions_human_readable)


if __name__ == "__main__":
    evaluate()
