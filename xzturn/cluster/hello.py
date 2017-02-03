import tensorflow as tf
import os

c = tf.constant("Hello, distributed TensorFlow!")
server = tf.train.Server.create_local_server()
sess = tf.Session(server.target)  # Create a session on the server.
print sess.run(c)

h, t = os.path.split("~/workspace/data/tmp.json")
print(t)
