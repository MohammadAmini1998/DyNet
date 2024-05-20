#!/usr/bin/env python
# coding=utf8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import logging
import make_env
import agents
import config
import time
import random
import tensorflow as tf
import numpy as np
import Environment_marl

FLAGS = config.flags.FLAGS
up_lanes = [i/2.0 for i in [3.5/2,3.5/2 + 3.5,250+3.5/2, 250+3.5+3.5/2, 500+3.5/2, 500+3.5+3.5/2]]
down_lanes = [i/2.0 for i in [250-3.5-3.5/2,250-3.5/2,500-3.5-3.5/2,500-3.5/2,750-3.5-3.5/2,750-3.5/2]]
left_lanes = [i/2.0 for i in [3.5/2,3.5/2 + 3.5,433+3.5/2, 433+3.5+3.5/2, 866+3.5/2, 866+3.5+3.5/2]]
right_lanes = [i/2.0 for i in [433-3.5-3.5/2,433-3.5/2,866-3.5-3.5/2,866-3.5/2,1299-3.5-3.5/2,1299-3.5/2]]

width = 750/2
height = 1298/2
n_veh = 4
n_neighbor = 1
n_RB = n_veh

env = Environment_marl.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, n_neighbor)
env.new_random_game()


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    return None


if __name__ == '__main__':

    set_seed(1)

    # logger_env = logging.getLogger('GridMARL')
    # logger_agent = logging.getLogger('Agent')

    # env = make_env.make_env(FLAGS.scenario)
    # logger_env.info('GridMARL Start with %d predator(s) and %d prey(s)', FLAGS.n_predator, FLAGS.n_prey)

    # logger_agent.info('Agent: {}'.format(FLAGS.agent))
    trainer = agents.load(FLAGS.agent+"/trainer.py").Trainer(env)

    print(FLAGS.agent, config.file_name)

    if FLAGS.train:
        start_time = time.time()
        trainer.learn()
        finish_time = time.time()
        trainer.test()
        print("TRAINING TIME (sec)", finish_time - start_time)
    else:
        trainer.test()


    print("LOG_FILE:\t" + config.log_filename)