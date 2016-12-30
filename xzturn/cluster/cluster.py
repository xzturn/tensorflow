from __future__ import print_function

import tensorflow as tf

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS

# Now we build a simple TF cluster
#   Master: 10.103.41.141 (CPU)
#   Worker1: 10.183.31.130 (GPU with 2 cards)
#   Worker2: 10.183.30.142 (GPU with 2 cards)


def main(_):
    cluster = tf.train.ClusterSpec({
        "worker": [
            "10.183.31.130:2238",  # /job:worker/task:0
            "10.183.31.130:2239",  # /job:worker/task:1
            "10.183.31.142:2238"   # /job:worker/task:2
            "10.183.31.142:2239"   # /job:worker/task:3
        ],
        "ps": [
            "10.103.41.141:2237"   # /job:ps/task:0
        ]
    })

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        print("PS: {}".format(server))
        server.join()
    elif FLAGS.job_name == "worker":
        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):
            # Build Model ...
            loss = tf.constant(0.0)    # not real
            global_step = tf.Variable(0)
            train_op = tf.train.AdagradOptimizer(0.01).minimize(loss, global_step=global_step)
            saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()
        # Create a supervisor which oversees the training process
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 logdir=".",
                                 init_op=init_op,
                                 summary_op=summary_op,
                                 saver=saver,
                                 global_step=global_step,
                                 save_model_secs=600)
        # The supervisor takes care of session initialization, restoring from a checkpoint,
        # and closing when done or an error occurs
        with sv.managed_session(server.target) as sess:
            print("/job:worker/task:%d" % FLAGS.task_index)
            print(sess.run([train_op, global_step]))
        # Ask for all the services to stop
        sv.stop()


if __name__ == 'main':
    tf.app.run()
