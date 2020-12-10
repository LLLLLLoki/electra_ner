import tensorflow as tf
import numpy as np

a = tf.constant([[1,2,3,4,0,0],[5,6,7,8,0,0],[9,10,11,12,0,0]],dtype = tf.float32)
b = tf.constant([[1,2,3,4,0,0],[5,6,7,8,1,0],[9,10,11,12,1,0]],dtype = tf.float32)
d = tf.constant([[1,2,3,4,1,0],[5,6,7,8,1,0],[9,10,12,12,2,0]],dtype = tf.float32)
weights = tf.sequence_mask(np.array([4,4,6]))
weights = tf.ones_like(weights, tf.int32)
with tf.Session() as sess:
    print(sess.run(weights))

