# TF code scaffolding for building simple models.

import tensorflow as tf


class TFBaseModel(object):
    """
    generic(simple) model using tensorflow
    """

    def __init__(self):
        # initialize variables/model parameters
        pass

    def __del__(self):
        pass

    def _inference(self, X):
        # compute inference model over data X and return the result
        return

    def _loss(self, X, Y):
        # compute loss over training data X and expected values Y
        return 0

    def _inputs(self):
        # read/generate input training data X and expected outputs Y
        return None, None

    def _train(self, total_loss):
        # train / adjust model parameters according to computed total loss
        return None

    def _evaluate(self, sess, X, Y):
        # evaluate the resulting trained model
        return

    def train(self):
        # Launch the graph in a session, setup boilerplate
        with tf.Session() as sess:

            tf.initialize_all_variables().run()

            X, Y = self._inputs()

            total_loss = self._loss(X, Y)
            train_op = self._train(total_loss)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # actual training loop
            training_steps = 1000
            for step in range(training_steps):
                sess.run([train_op])
                # for debugging and learning purposes, see how the loss gets decremented thru training steps
                if step % 10 == 0:
                    print "loss: ", sess.run([total_loss])

            self._evaluate(sess, X, Y)

            coord.request_stop()
            coord.join(threads)
            sess.close()
