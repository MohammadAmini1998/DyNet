from __future__ import print_function, division, absolute_import

import numpy as np
from agents.schednet.agent import PredatorAgent
from agents.simple_agent import RandomAgent
from agents.evaluation import Evaluation
import logging
import config
from envs.gui import canvas
from torch.utils.tensorboard import SummaryWriter
import os 
import tensorflow as tf
from numba import jit

writer = SummaryWriter('logdir')

FLAGS = config.flags.FLAGS
logger = logging.getLogger('Agent')
result = logging.getLogger('Result')

training_step = FLAGS.training_step
testing_step = FLAGS.testing_step

epsilon_dec = 1/training_step
epsilon_min = 0.01

summary_writer = tf.summary.create_file_writer("logdir")


class Trainer(object):

    def __init__(self, env):
        logger.info("SchedNet trainer is created")

        self._env = env
        self._eval = Evaluation()
        self._agent_profile = self._env.get_agent_profile()
        self._n_predator = self._agent_profile['predator']['n_agent']
        self._n_prey = self._agent_profile['prey']['n_agent']
        
        # State and obs additionally include history information
        self._state_dim = self._env.get_info()[0]['state'].shape[0] 
        self._obs_dim = obs_dim=self._agent_profile['predator']['obs_dim'][0]
        
        # Predator agent
        self._predator_agent = PredatorAgent(n_agent=self._agent_profile['predator']['n_agent'],
                                             action_dim=self._agent_profile['predator']['act_dim'],
                                             state_dim=self._state_dim,
                                             obs_dim=self._obs_dim)
        # Prey agent (randomly moving)
        self._prey_agent = []
        for _ in range(self._n_prey):
            self._prey_agent.append(RandomAgent(5))

        self.epsilon = .5  # Init value for epsilon

        if FLAGS.gui:  # Enable GUI
            self.canvas = canvas.Canvas(self._n_predator, 1, FLAGS.map_size)
            self.canvas.setup()
    # Main function of the algorithm. 
    def learn(self):

        global_step = 0
        episode_num = 0
        print_flag = True

        while global_step < training_step:
            episode_num += 1
            step_in_ep = 0

            obs_n = self._env.reset()  
            info_n = self._env.get_info()
            state=info_n[0]['state']
            total_reward = 0
            done = False

            while not done:
                global_step += 1
                step_in_ep += 1

                action_n = self.get_action(obs_n, global_step)
                obs_n_next, reward_n, done_n, info_n = self._env.step(action_n)
                state_next=info_n[0]['state']
                
                if FLAGS.gui:
                    self.canvas.draw(state_next * FLAGS.map_size, [0]*self._n_predator, "Train")

                done_single = sum(done_n) > 0
                self.train_agents(state, obs_n, action_n, reward_n, state_next, obs_n_next, done_single)
                # self._env.render()
                
                obs_n = obs_n_next
                state = state_next
                total_reward += np.sum(reward_n)
                

                with summary_writer.as_default():
                    if is_episode_done(done_n, global_step):
                        if FLAGS.gui:
                            self.canvas.draw(state_next * FLAGS.map_size, [0]*self._n_predator, "Train", True)
                        if print_flag:
                            print("[train_ep %d]" % (episode_num),"\tstep:", global_step, "\tstep_per_ep:", step_in_ep, "\treward", total_reward,"\tepsilon",self.epsilon)
                        done = True
        
                    writer.add_scalar("Reward in each episode", total_reward, global_step)
                    writer.add_scalar("Total step in each episode", step_in_ep, global_step)

                if FLAGS.eval_on_train and global_step % FLAGS.eval_step == 0:
                    self.test(global_step)
                    break

        self._predator_agent.save_nn(global_step)
        self._eval.summarize()

    def get_action(self, obs_n, global_step, train=True):

        act_n = [0] * len(obs_n)
        self.epsilon = max(self.epsilon - epsilon_dec, epsilon_min)

        # Action of predator
        if train and (global_step < FLAGS.m_size * FLAGS.pre_train_step or np.random.rand() < self.epsilon):  # with prob. epsilon
            # Exploration
            predator_action = self._predator_agent.explore()
        else:
            # Exploitation
            predator_obs = [obs_n[i] for i in self._agent_profile['predator']['idx']]
            predator_action = self._predator_agent.act(predator_obs)

        for i, idx in enumerate(self._agent_profile['predator']['idx']):
            act_n[idx] = predator_action[i]

        # Action of prey
        for i, idx in enumerate(self._agent_profile['prey']['idx']):
            act_n[idx] = self._prey_agent[i].act(None)

        return np.array(act_n, dtype=np.int32)


    def train_agents(self, state, obs_n, action_n, reward_n, state_next, obs_n_next, done):
        
        predator_obs = [obs_n[i] for i in self._agent_profile['predator']['idx']]
        predator_action = [action_n[i] for i in self._agent_profile['predator']['idx']]
        predator_reward = [reward_n[i] for i in self._agent_profile['predator']['idx']]
        predator_obs_next = [obs_n_next[i] for i in self._agent_profile['predator']['idx']]
        self._predator_agent.train(state, predator_obs, predator_action, predator_reward,
                                   state_next, predator_obs_next, done)
    #  The method concatenates the observations of the predator agents with the
    #  communication schedule history and returns the modified observation and state arrays.
    

    #  method and concatenates the observations of the predator
    #  agents with the updated communication schedule history to form the
    #  modified observation array. The method also concatenates the state of the environment 
    #  with the communication schedule history to form the modified state array.

    

    #The update_h_schedule() method takes as input the current communication schedule history (h_schedule) 
    # and the communication schedule for the current step (schedule_n).
    
    def print_obs(self, obs):
        for i in range(FLAGS.n_predator):
            print(obs[i])
        print("")

    def check_obs(self, obs):

        check_list = []
        for i in range(FLAGS.n_predator):
            check_list.append(obs[i][2])

        return np.array(check_list)
    
    def test(self, curr_ep=None):

        global_step = 0
        episode_num = 0

        total_reward = 0
        obs_cnt = np.zeros(self._n_predator)
        

        while global_step < testing_step:
            episode_num += 1
            step_in_ep = 0
            obs_n = self._env.reset()  
            info_n = self._env.get_info()
            state=info_n[0]['state']

            while True:

                global_step += 1
                step_in_ep += 1

                action_n = self.get_action(obs_n, global_step, False)
                obs_n_next, reward_n, done_n, info_n = self._env.step(action_n)
                state_next=info_n[0]['state']
                obs_cnt += self.check_obs(obs_n_next)

                if FLAGS.gui:
                    self.canvas.draw(state_next * FLAGS.map_size, [0]*self._n_predator, "Test")

                obs_n = obs_n_next
                state = state_next
                total_reward += np.sum(reward_n)

                if is_episode_done(done_n, global_step, "test") or step_in_ep > FLAGS.max_step:
                    if FLAGS.gui:
                        self.canvas.draw(state_next * FLAGS.map_size, [0]*self._n_predator, "Test", True)
                    break

        print("Test result: Average steps to capture: ", curr_ep, float(global_step) / episode_num,
              "\t", float(total_reward) / episode_num, obs_cnt / episode_num)
        self._eval.update_value("test_result", float(global_step)/episode_num, curr_ep)


def is_episode_done(done, step, e_type="train"):

    if e_type == "test":
        if sum(done) > 0 or step >= FLAGS.testing_step:
            return True
        else:
            return False

    else:
        if sum(done) > 0 or step >= FLAGS.training_step:
            return True
        else:
            return False


