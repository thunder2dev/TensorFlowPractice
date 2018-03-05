import tensorflow as tf
# hello = tf.constant('Hello, TensorFlow!')
weight = tf.Variable(tf.random_normal([2, 3], stddev=2, seed=1), name="w1")
x = tf.constant([[0.7, 0.9]])

a = tf.matmul(x, weight)

weight2 = tf.Variable(tf.random_normal([2, 3], stddev=2, seed=2), name="w2")

weight.assign(weight2)


sess = tf.Session()
sess.run(weight.initializer)
sess.run(weight2.initializer)

print(sess.run(weight))
print(sess.run(weight2))
print(sess.run(weight2))
print(sess.run(a))
