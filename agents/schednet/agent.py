# coding=utf8

from __future__ import print_function, division, absolute_import
import random
import numpy as np
import tensorflow as tf
import math 
from agents.schednet.replay_buffer import ReplayBuffer
from agents.schednet.ac_network import ActionSelectorNetwork
from agents.schednet.ac_network import CriticNetwork
from agents.schednet.sched_network import WeightGeneratorNetwork
from agents.schednet.bit_network import WeightGeneratorNetwork1
from agents.schednet.scheduler import Scheduler
from agents.evaluation import Evaluation

import logging
import config

FLAGS = config.flags.FLAGS
logger = logging.getLogger('Agent')
result = logging.getLogger('Result')


class PredatorAgent(object):

    def __init__(self, n_agent, action_dim, state_dim, obs_dim, name=""):
        logger.info("Predator Agent is created")

        self._n_agent = n_agent
        self._state_dim = state_dim
        self._action_dim_per_unit = action_dim
        self._obs_dim_per_unit = obs_dim
        self._obs_dim = self._obs_dim_per_unit * self._n_agent

        self._name = name
        self.update_cnt = 0

        # Make Networks
        tf.compat.v1.reset_default_graph()
        my_graph = tf.Graph()

        with my_graph.as_default():
            self.sess = tf.compat.v1.Session(graph=my_graph, config=tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)))

            self.action_selector = ActionSelectorNetwork(self.sess, self._n_agent, self._obs_dim_per_unit, self._action_dim_per_unit, self._name)
            self.weight_generator = WeightGeneratorNetwork(self.sess, self._n_agent, self._obs_dim)

            self.critic = CriticNetwork(self.sess, self._n_agent, self._state_dim, self._name)
            self.scheduler=Scheduler(self.sess, self._n_agent,  self._n_agent, self._name)
            tf.compat.v1.global_variables_initializer().run(session=self.sess)
            self.saver = tf.compat.v1.train.Saver()

            if FLAGS.load_nn:
                if FLAGS.nn_file == "":
                    logger.error("No file for loading Neural Network parameter")
                    exit()
                self.saver.restore(self.sess, FLAGS.nn_file)

        self.replay_buffer = ReplayBuffer()
        self._eval = Evaluation()

    def save_nn(self, global_step):
        self.saver.save(self.sess, config.nn_filename, global_step)
    # The act() method takes as input a list of observations (obs_list) and a list of communication schedules 
    # (schedule_list). The method first concatenates the observations of all predator agents into a single observation
    #  tensor and passes it to the ActionSelectorNetwork to obtain the probability distribution over actions for each predator 
    # agent. The method then samples an action for
    #  each predator agent using the corresponding probability distribution and returns the list of actions.
    
    def act(self, obs_list, schedule_list):
        c_new=self.convert(schedule_list,FLAGS.capa)
        c_new=np.array([c_new])
        count = np.count_nonzero(c_new)
        count=np.array([count])
        count=np.reshape(count,[-1,1])
        action_prob_list = self.action_selector.action_for_state(np.concatenate(obs_list)
                                                                   .reshape(1, self._obs_dim),c_new,count)

        if np.isnan(action_prob_list).any():
            raise ValueError('action_prob contains NaN')

        action_list = []
        for action_prob in action_prob_list.reshape(self._n_agent, self._action_dim_per_unit):
            action_list.append(np.random.choice(len(action_prob), p=action_prob))

        return action_list,count

    def train(self, state,action, obs_list, action_list, reward_list, state_next, obs_next_list, schedule_n,count, priority, done):
        s = state
        action=action
        count=count
        o = obs_list
        a = action_list
        r = np.sum(reward_list)
        s_ = state_next
        o_ = obs_next_list
        c = schedule_n
        p = priority

        self.store_sample(s,action, o, a, r, s_, o_, c,count, p, done)
        self.update_ac()
        return 0

    def store_sample(self, s,action, o, a, r, s_, o_, c,count, p, done):
        c_new=self.convert(c,FLAGS.capa)

        self.replay_buffer.add_to_memory((s,action, o, a, r, s_, o_, c,c_new,count, p, done))
        return 0
    def convert(self,c,capacity):
        result = []
        for num in c:
            num=int(num)
            if num == 0:
                result.extend([False] * capacity)
            else:
                result.extend([True] * min(num, capacity))
                result.extend([False] * (capacity - num))
        return result

        

    
    def update_ac(self):
        
        if len(self.replay_buffer.replay_memory) < FLAGS.pre_train_step * FLAGS.m_size:
            return 0

        minibatch = self.replay_buffer.sample_from_memory()
        s,action, o, a, r, s_, o_, c,c_new,count, p, d = map(np.array, zip(*minibatch))
        o = np.reshape(o, [-1, self._obs_dim])
        o_ = np.reshape(o_, [-1, self._obs_dim])
        p = np.reshape(p, [-1, self._n_agent])
        action=np.reshape(action,[-1,1])
        count=np.reshape(count,[-1,1])
        # r=r.reshape(-1,1)
        
        # r1=r1.reshape(-1)

        p_ = self.weight_generator.target_schedule_for_obs(o_)

        p_ = np.reshape(p_, [-1, self._n_agent])


        scheduler_input=np.reshape(p, [-1, 4])
        scheduler_next_input=np.reshape(p_, [-1, 4])
        
        
        # next_state_com=tf.concat([p_,p_1],axis=-1)

        
    
        td_error, _ = self.critic.training_critic(o, r, o_, p, p_, d)  # train critic'
        
        
        _ = self.action_selector.training_actor(o, a, c_new,count, td_error)  # train actor

        wg_grads = self.critic.grads_for_scheduler(o, p)


        _ = self.weight_generator.training_weight_generator(o, wg_grads)
        _ = self.critic.training_target_critic()  # train slow target critic
        _ = self.weight_generator.training_target_weight_generator()



        _=self.scheduler.training_critic(action,scheduler_input, r, scheduler_next_input, d)
        _ = self.scheduler.training_target_critic()

        return 0

    def schedule(self, obs_list,l,epsilon,action_space):
        priority = self.weight_generator.schedule_for_obs(np.concatenate(obs_list)
                                                           .reshape(1, self._obs_dim))
        
        obs_list=np.concatenate(obs_list).reshape(1, self._obs_dim)
        noise = np.random.normal(loc=0, scale=.3, size=priority.shape)

        # Add noise to priority and priority1
        priority_with_noise = priority + noise

        # Clip values in priority and priority1
        priority_with_noise = np.clip(priority_with_noise, 0, 1)

        # Update priority and priority1 with the clipped versions
        priority = priority_with_noise

        if np.random.rand()<epsilon:

            action = np.random.randint(0, 20)
            c_new=action_space[action]
      
        else:
            input=np.concatenate([priority],axis=-1)
            input=np.reshape(input,[-1,4])
            bandwidth=self.scheduler.get_com_action(input)
            action = np.argmax(bandwidth)
            c_new=action_space[action]


    
        return c_new,action, priority
    def explore(self):
        return [random.randrange(self._action_dim_per_unit)
                for _ in range(self._n_agent)]


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)
