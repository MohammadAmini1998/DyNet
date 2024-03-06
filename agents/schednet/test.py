import tensorflow as tf
threshold=.5
ret = 1.3
binary_output = tf.cast(tf.math.greater(ret, threshold), tf.float32)

print(binary_output)
