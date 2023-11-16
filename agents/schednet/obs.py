from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import tensorflow as tf
import numpy as np
import config
import sys
import torch 

FLAGS = config.flags.FLAGS


# The function first generates an encoder network for each predator agent's observation 
# using the encoder_network() function. It then generates a decoder network that takes in the encoded observations
# of all predator agents and the communication schedule for the current step and generates a concatenated 
# communication message for all predator agents using the decode_concat_network() function.

def generate_comm_network(obs_list, obs_dim_per_unit, action_dim, n_agent, trainable=True, share=False):
    actions = list()
    h_num = 32

    capacity = FLAGS.capa 

    # Generate encoder
    encoder_scope = "encoder"
    aggr_scope = "aggr"
    decoder_out_dim = 16
    encoder_list = list()

    # Generate actor
    scope = "comm"
    for i in range(n_agent):
        if not FLAGS.s_share:
            scope = "comm" + str(i)

        with tf.compat.v1.variable_scope(scope):
            obs_weight = weight_generator(obs_list[i], obs_dim_per_unit, h_num, trainable)

        actions.append(obs_weight)

    return tf.concat(actions, axis=-1)



# Action selector: 
def weight_generator(obs,obs_dim_per_unit, h_num, trainable=True):
    
    
    hidden_1 = tf.keras.layers.Dense(units=h_num, activation=tf.nn.relu,
                                     kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                                     bias_initializer=tf.constant_initializer(0.1),  # biases
                                     use_bias=True, trainable=trainable, name='sender_1')(obs)
    hidden_2 = tf.keras.layers.Dense(units=h_num, activation=tf.nn.relu,
                                     kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                                     bias_initializer=tf.constant_initializer(0.1),  # biases
                                     use_bias=True, trainable=trainable, name='sender_2')(hidden_1)

    hidden_3 = tf.keras.layers.Dense(units=h_num, activation=tf.nn.relu,
                                     kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                                     bias_initializer=tf.constant_initializer(0.1),  # biases
                                     use_bias=True, trainable=trainable, name='sender_3')(hidden_2)

    a = tf.keras.layers.Dense(units=obs_dim_per_unit, activation=tf.nn.sigmoid,
                              kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                              bias_initializer=tf.constant_initializer(0.1),  # biases
                              use_bias=True, trainable=trainable, name='sender_4')(hidden_3)
    return a



