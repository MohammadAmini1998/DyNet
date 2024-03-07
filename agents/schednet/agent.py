# coding=utf8

from __future__ import print_function, division, absolute_import
import random
import numpy as np
import tensorflow as tf

from agents.schednet.replay_buffer import ReplayBuffer
from agents.schednet.ac_network import ActionSelectorNetwork
from agents.schednet.ac_network import CriticNetwork
from agents.schednet.sched_network import WeightGeneratorNetwork
from agents.evaluation import Evaluation

import logging
import config

FLAGS = config.flags.FLAGS
logger = logging.getLogger('Agent')
result = logging.getLogger('Result')


class PredatorAgent(object):

    def __init__(self, n_agent, action_dim, state_dim, obs_dim, name=""):
        logger.info("Predator Agent is created")
        self.capacity=FLAGS.capa
        self._n_agent = n_agent
        self._state_dim = state_dim
        self._action_dim_per_unit = action_dim
        self._obs_dim_per_unit = obs_dim
        self._obs_dim = self._obs_dim_per_unit * self._n_agent

        self._name = name
        self.update_cnt = 0
        self.capacity=FLAGS.capa
        # Make Networks
        tf.compat.v1.reset_default_graph()
        my_graph = tf.Graph()

        with my_graph.as_default():
            self.sess = tf.compat.v1.Session(graph=my_graph, config=tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)))

            self.action_selector = ActionSelectorNetwork(self.sess, self._n_agent, self._obs_dim_per_unit, self._action_dim_per_unit, self._name)
            self.weight_generator = WeightGeneratorNetwork(self.sess, self._n_agent, self._obs_dim)
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
    def convert(self, allocation,capacity):
        allocation_mask=[]
        allocation=list(allocation)
        for i in allocation:
            my_list_true = [True for _ in range(int(i))]
            my_list_False=[False for _ in range(int(capacity-i))]
            allocation_mask.extend(my_list_true)
            allocation_mask.extend(my_list_False)
        count_true = allocation_mask.count(True)
        return allocation_mask

    def act(self, obs_list, schedule_list,priority):
        bandwidth=self.bandwidth_allocation(priority)
        bandwidth=list(bandwidth)

        bandwidth=self.convert(bandwidth,FLAGS.capa)
        bandwidth=np.array(bandwidth,dtype=np.float32)
        action_prob_list = self.action_selector.action_for_state(np.concatenate(obs_list)
                                                                   .reshape(1, self._obs_dim),
                                                                 schedule_list.reshape(1, self._n_agent),
                                                                 bandwidth.reshape(1,self.capacity * self._n_agent))
        

        if np.isnan(action_prob_list).any():
            raise ValueError('action_prob contains NaN')

        action_list = []
        for action_prob in action_prob_list.reshape(self._n_agent, self._action_dim_per_unit):
            action_list.append(np.random.choice(len(action_prob), p=action_prob))

        return action_list
    
    def bandwidth_allocation(self,weights,capacity=FLAGS.capa):
        softmax_weights = np.exp(weights) / np.sum(np.exp(weights))

        # allocation=np.array(allocation,dtype=np.float32)
        allocation=softmax_weights*capacity
        allocation=np.round(allocation)
        if np.sum(allocation)<capacity:
            while np.sum(allocation)<capacity:
                max_value = np.argmax(allocation)
                allocation[max_value]+=1

        if np.sum(allocation)>capacity:
            while np.sum(allocation)>capacity:
                max_value = np.argmin(allocation)
                allocation[max_value]-=1

        allocation=allocation.tolist()
        # print(allocation)
        # print(allocation)
        return allocation

    def train(self, state, obs_list, action_list, reward_list, state_next, obs_next_list, schedule_n, priority, done):

        s = state
        o = obs_list
        a = action_list
        r = np.sum(reward_list)
        s_ = state_next
        o_ = obs_next_list
        c = schedule_n
        p = priority
        bandwidth=self.bandwidth_allocation(priority)

        bandwidth_mask=self.convert(list(bandwidth),self.capacity)
        self.store_sample(s, o, a, r, s_, o_, c, p,bandwidth,bandwidth_mask, done)
        self.update_ac()
        return 0

    def store_sample(self, s, o, a, r, s_, o_, c, p,bandwidth,bandwidth_mask, done):
        self.replay_buffer.add_to_memory((s, o, a, r, s_, o_, c, p,bandwidth,bandwidth_mask, done))
        return 0

    def update_ac(self):
        
        if len(self.replay_buffer.replay_memory) < FLAGS.pre_train_step * FLAGS.m_size:
            return 0

        minibatch = self.replay_buffer.sample_from_memory()
        s, o, a, r, s_, o_, c, p,bandwidth,bandwidth_mask, d = map(np.array, zip(*minibatch))
        o = np.reshape(o, [-1, self._obs_dim])
        o_ = np.reshape(o_, [-1, self._obs_dim])

        p_ = self.weight_generator.target_schedule_for_obs(o_)
        
        td_error, _ = self.critic.training_critic(s, r, s_, p, p_, d)  # train critic
        _ = self.action_selector.training_actor(o, a, c,bandwidth_mask, td_error)  # train actor

        wg_grads = self.critic.grads_for_scheduler(s, p)
        _ = self.weight_generator.training_weight_generator(o, wg_grads)
        _ = self.critic.training_target_critic()  # train slow target critic
        _ = self.weight_generator.training_target_weight_generator()

        return 0

    def schedule(self, obs_list):
        priority = self.weight_generator.schedule_for_obs(np.concatenate(obs_list)
                                                           .reshape(1, self._obs_dim))

        if FLAGS.sch_type == "top":
            schedule_idx = np.argsort(-priority)[:FLAGS.s_num]
        elif FLAGS.sch_type == "softmax":
            sm = softmax(priority)
            schedule_idx = np.random.choice(self._n_agent, p=sm)
        else: # IF N_SUM == 1
            schedule_idx = np.argmax(priority)
                            
        ret = np.zeros(self._n_agent)
        ret[schedule_idx] = 1.0
        return ret, priority

    def explore(self):
        return [random.randrange(self._action_dim_per_unit)
                for _ in range(self._n_agent)]


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)
