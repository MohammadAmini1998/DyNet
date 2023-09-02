import tensorflow as tf

x = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)

mean = tf.reduce_mean(x)  # Result: 3.5

print(mean)