import tensorflow as tf
sess = tf.compat.v1.Session()
gpu_available = tf.config.list_physical_devices('GPU')
print("CUDA is available: ", gpu_available)
