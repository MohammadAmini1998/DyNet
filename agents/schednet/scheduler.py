 #!/usr/bin/env python
# coding=utf8

import numpy as np
import tensorflow as tf
import config
from agents.schednet import comm

FLAGS = config.flags.FLAGS

h1_scheduler = 32  # hidden layer 1 size 
h2_scheduler = 32  # hidden layer 2 size 
lr_wg = FLAGS.w_lr   # learning rate for the weight generator
lr_decay = 1  # learning rate decay (per episode)
tau = 5e-2 # soft target update rate


class Scheduler:
    def __init__(self, sess, n_player, obs_dim):

        self.sess = sess
        self.obs_dim = obs_dim # concatenated observation space
        self.n_player = n_player

        # placeholders
        self.weights_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.n_player])
        self.td_errors = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 1])
        self.is_training_ph = tf.compat.v1.placeholder(dtype=tf.bool, shape=())  # for dropout

        self.sched_grads_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 4])

        with tf.compat.v1.variable_scope('scheduler'):
            self.schedule_policy = self.generate_wg(self.weights_ph, trainable=True)

        with tf.compat.v1.variable_scope('slow_target_scheduler'):
            self.target_schedule_policy = tf.stop_gradient(self.generate_wg(self.weights_ph, trainable=False))

        self.actor_vars = tf.compat.v1.get_collection(tf.compat.v1.trainable_variables, scope='scheduler')
        var_grads = tf.gradients(self.schedule_policy, self.actor_vars, -self.sched_grads_ph)
        self.scheduler_train_op = tf.optimizers.Adam(lr_wg * lr_decay).apply_gradients(
            zip(var_grads, self.actor_vars))

        slow_target_sch_vars = tf.compat.v1.get_collection(tf.compat.v1.trainable_variables, scope='slow_target_scheduler')

        # update values for slowly-changing targets towards current actor and critic
        update_slow_target_ops_i = []
        for i, slow_target_sch_var in enumerate(slow_target_sch_vars):
            update_slow_target_sch_op = slow_target_sch_var.assign(
                tau * self.actor_vars[i] + (1 - tau) * slow_target_sch_var)
            update_slow_target_ops_i.append(update_slow_target_sch_op)
        self.update_slow_targets_op_i = tf.group(*update_slow_target_ops_i)

    def generate_wg(self, weights, trainable=True):
        s = self.generate_wg_network(weights, trainable)
        return s

    def generate_wg_network(self, weights, trainable=True):
        hidden_1 = tf.compat.v1.layers.dense(weights, h1_scheduler, activation=tf.nn.relu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),
                                   bias_initializer=tf.constant_initializer(0.1),  
                                   use_bias=True, trainable=trainable)

        hidden_2 = tf.compat.v1.layers.dense(hidden_1, h2_scheduler, activation=tf.nn.relu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),
                                   bias_initializer=tf.constant_initializer(0.1),  
                                   use_bias=True, trainable=trainable)

        schedule = tf.compat.v1.layers.dense(hidden_2, self.n_player, activation=tf.nn.sigmoid, trainable=trainable)

        return schedule

    def schedule_for_obs(self, weights_ph):

        return self.sess.run(self.schedule_policy,
                             feed_dict={self.weights_ph: weights_ph, self.is_training_ph: False})[0]

    def target_schedule_for_obs(self, weights_ph):

        return self.sess.run(self.target_schedule_policy,
                             feed_dict={self.weights_ph: weights_ph, self.is_training_ph: False})

    def training_weight_generator(self, weights_ph, sched_grads_ph):

        return self.sess.run(self.scheduler_train_op,
                             feed_dict={self.weights_ph: weights_ph,
                                        self.sched_grads_ph: sched_grads_ph,
                                        self.is_training_ph: True})

    def training_target_weight_generator(self):
        return self.sess.run(self.update_slow_targets_op_i)
