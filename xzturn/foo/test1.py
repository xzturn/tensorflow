from __future__ import print_function

import tensorflow as tf

a = tf.constant(5, name='input_a')
b = tf.constant(3, name='input_b')
c = tf.mul(a, b, name='mul_c')
d = tf.add(a, b, name='add_d')
e = tf.add(c, d, name='add_e')

with tf.Session() as sess:
    output = sess.run(e)
    writer = tf.train.SummaryWriter('./tbtmp', sess.graph)
    print(output)
    writer.close()
