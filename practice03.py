import tensorflow as tf

v = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

v1 = tf.constant([[1.0, 2.0], [2.0, 1.0]])
v2 = tf.constant(([3.0, 4.0], [4.0, 3.0]))

# v = tf.get_variable("v", shape=[1], initializer=tf.constant_initializer(1.0))

# with tf.variable_scope("foo"):
#    v = tf.get_variable("v", [1], initializer=tf.constant_initializer(1.0))

dotMul = v1 * v2


y1 = tf.constant([0.1, 0.2, 0.8])
y2 = tf.constant([0, 0, 1])

# cross = tf.nn.softmax_cross_entropy_with_logits(y, y_)
# sparse_softmax_cross_entropy_with_logits(y, tf.argmax(y_, 1))
# mse = tf.reduce_mean(tf.square(y_ - y)
# loss = tf.reduce_sum(tf.select(tf.greater(v1, v2), (v1-v2)*a, (v2-v1)*b)



sess = tf.Session()
with sess.as_default():
    print(tf.clip_by_value(v, 2.1, 4.1).eval())
    print(tf.log(v).eval())
    print(dotMul.eval())
    print(cross.eval())
