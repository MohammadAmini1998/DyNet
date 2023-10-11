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

            self.weight_generator1 = WeightGeneratorNetwork1(self.sess, self._n_agent, self._obs_dim)
            self.critic = CriticNetwork(self.sess, self._n_agent, self._state_dim, self._name)

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
        action_prob_list = self.action_selector.action_for_state(np.concatenate(obs_list)
                                                                   .reshape(1, self._obs_dim),c_new)

        if np.isnan(action_prob_list).any():
            raise ValueError('action_prob contains NaN')

        action_list = []
        for action_prob in action_prob_list.reshape(self._n_agent, self._action_dim_per_unit):
            action_list.append(np.random.choice(len(action_prob), p=action_prob))

        return action_list

    def train(self, state, obs_list, action_list, reward_list, state_next, obs_next_list, schedule_n, priority,priority1, done):

        s = state
        
        o = obs_list
        a = action_list
        r = np.sum(reward_list)
        s_ = state_next
        o_ = obs_next_list
        c = schedule_n
        p = priority
        p1=priority1

        self.store_sample(s, o, a, r, s_, o_, c, p,p1, done)
        self.update_ac()
        return 0

    def store_sample(self, s, o, a, r, s_, o_, c, p,p1, done):
        # print(c)
        c_new=self.convert(c,FLAGS.capa)
        self.replay_buffer.add_to_memory((s, o, a, r, s_, o_, c,c_new, p,p1, done))
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
        s, o, a, r, s_, o_, c,c_new, p,p1, d = map(np.array, zip(*minibatch))
        o = np.reshape(o, [-1, self._obs_dim])
        o_ = np.reshape(o_, [-1, self._obs_dim])

        p = np.reshape(p, [-1, self._n_agent])
        
        p_ = self.weight_generator.target_schedule_for_obs(o_)

        p_ = np.reshape(p_, [-1, self._n_agent])

        p_1 = self.weight_generator1.target_schedule_for_obs(o_,p_)

        # msg=self.action_selector.get_messages(o,c_new)      
        # print(msg)  
        # print(c_new)
       

    
        
        td_error, _ = self.critic.training_critic(o, r, o_, p, p_,p1,p_1, d)  # train critic'
        

        _ = self.action_selector.training_actor(o, a, c_new, td_error)  # train actor

        wg_grads = self.critic.grads_for_scheduler(o, p)

        wg_grads1 = self.critic.grads_for_scheduler1(o, p1)

        _ = self.weight_generator.training_weight_generator(o, wg_grads)
        _ = self.critic.training_target_critic()  # train slow target critic
        _ = self.weight_generator.training_target_weight_generator()


        _ = self.weight_generator1.training_weight_generator(o,p , wg_grads1)
        _ = self.weight_generator1.training_target_weight_generator()

        return 0

    def schedule(self, obs_list,l):
        priority = self.weight_generator.schedule_for_obs(np.concatenate(obs_list)
                                                           .reshape(1, self._obs_dim))
        
        obs_list=np.concatenate(obs_list).reshape(1, self._obs_dim)
        priority1 = self.weight_generator1.schedule_for_obs(obs_list, priority.reshape(1,self._n_agent))

        # Sort the agents based on priority1 in descending order
        softmax_weights = np.exp(priority1) / np.sum(np.exp(priority1))
        allocation = np.zeros(self._n_agent)

        sorted_agents = np.argsort(-softmax_weights)
        # print(priority1)
        # print(sorted_agents)
        
        remaining_bandwidth = l
    # Allocate bandwidth to agents starting from the largest priority1 values
        while remaining_bandwidth>0:
            for agent in sorted_agents:
                    agent_allocation = np.ceil((softmax_weights[agent]*remaining_bandwidth))
                    allocation[agent] = agent_allocation
                    remaining_bandwidth -= agent_allocation
        # print(priority1)
        # print(allocation)
        

    # Adjust the allocation to ensure the total sum is equal to l
        # if np.sum(allocation) < l:
        #     remaining_allocation = l - np.sum(allocation)
        #     max_allocation_agent = np.argmax(allocation)
        #     allocation[max_allocation_agent] += remaining_allocation

        # # print(priority1)

        if FLAGS.sch_type == "top":
            schedule_idx = np.argsort(-priority1)[:FLAGS.s_num]
        elif FLAGS.sch_type == "softmax":
            sm = softmax(priority1)
            schedule_idx = np.random.choice(self._n_agent, p=sm)
        else: # IF N_SUM == 1
            schedule_idx = np.argmax(priority1)
                            
        ret = np.zeros(self._n_agent)
        ret[schedule_idx] = 1.0

        return allocation, priority, priority1

    def explore(self):
        return [random.randrange(self._action_dim_per_unit)
                for _ in range(self._n_agent)]


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)
