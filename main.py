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
from numba import cuda,jit

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


FLAGS = config.flags.FLAGS

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    return None

@jit(target_backend='cuda')   
def main():
    set_seed(1)

    logger_env = logging.getLogger('GridMARL')
    logger_agent = logging.getLogger('Agent')

    env = make_env.make_env(FLAGS.scenario)
    logger_env.info('GridMARL Start with %d predator(s) and %d prey(s)', FLAGS.n_predator, FLAGS.n_prey)

    logger_agent.info('Agent: {}'.format(FLAGS.agent))
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

if __name__ == '__main__':
    main()