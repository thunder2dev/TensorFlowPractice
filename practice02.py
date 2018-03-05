import tensorflow as tf

w = tf.Variable(tf.random_normal([3, 2], stddev=1))

x = tf.placeholder(tf.float32, shape=(1, 3), name="input")

y = tf.matmul(x, w)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(y, feed_dict={x: [[1, 2, 3]]}))




















