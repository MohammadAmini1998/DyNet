from __future__ import print_function, division, absolute_import
import math 

import numpy as np
from agents.schednet.agent import PredatorAgent
from agents.simple_agent import RandomAgent
from agents.evaluation import Evaluation
import logging
import config
from envs.gui import canvas
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf

FLAGS = config.flags.FLAGS
logger = logging.getLogger('Agent')
result = logging.getLogger('Result')

training_step = FLAGS.training_step
testing_step = FLAGS.testing_step
step1=750000
epsilon_dec = 1/750000
epsilon_min = 0.01

summary_writer = tf.summary.create_file_writer("logdir")
writer = SummaryWriter()

class Trainer(object):
  
    def __init__(self, env):
        logger.info("SchedNet trainer is created")

        self._env = env
        self._eval = Evaluation()
        self._agent_profile = self._env.get_agent_profile()
        self._n_predator = self._agent_profile['predator']['n_agent']
        self._n_prey = self._agent_profile['prey']['n_agent']
        
        # State and obs additionally include history information
        self._state_dim = self._env.get_info()[0]['state'].shape[0] + self._n_predator

        self._obs_dim = obs_dim=self._agent_profile['predator']['obs_dim'][0] + 1
        
        # Predator agent
        self._predator_agent = PredatorAgent(n_agent=self._agent_profile['predator']['n_agent'],
                                             action_dim=self._agent_profile['predator']['act_dim'],
                                             state_dim=self._state_dim,
                                             obs_dim=self._obs_dim)
        # Prey agent (randomly moving)
        self._prey_agent = []
        for _ in range(self._n_prey):
            self._prey_agent.append(RandomAgent(5))

        self.epsilon = .5 # Init value for epsilon

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
            h_schedule_n = np.zeros(self._n_predator)  # schedule history
            obs_n, state, _ = self.get_obs_state_with_schedule(obs_n, info_n, h_schedule_n, init=True)
           

            total_reward = 0
            done = False

            while not done:
                global_step += 1
                step_in_ep += 1

                schedule_n, priority,priority1 = self.get_schedule(obs_n, global_step, FLAGS.sched)
                action_n,probs = self.get_action(obs_n,priority1, schedule_n, global_step)
                obs_n_without_schedule, reward_n, done_n, info_n = self._env.step(action_n)
                obs_n_next, state_next, h_schedule_n = self.get_obs_state_with_schedule(obs_n_without_schedule, info_n, h_schedule_n, schedule_n)
                if FLAGS.gui:
                    self.canvas.draw(state_next * FLAGS.map_size, [0]*self._n_predator, "Train")

                done_single = sum(done_n) > 0
                self.train_agents(state, obs_n, action_n,probs, reward_n, state_next, obs_n_next, schedule_n, priority,priority1, done_single,global_step)
                
                
                obs_n = obs_n_next
                state = state_next
                total_reward += np.sum(reward_n)


                if global_step%10000==0:
                    print(priority)
                    print(priority1)
                

                with summary_writer.as_default():
                    if is_episode_done(done_n, global_step):
                        if FLAGS.gui:
                            self.canvas.draw(state_next * FLAGS.map_size, [0]*self._n_predator, "Train", True)
                        if print_flag:
                            print("[train_ep %d]" % (episode_num),"\tstep:", global_step, "\tstep_per_ep:", step_in_ep, "\treward", total_reward,"\tepsilon",self.epsilon)
                        done = True
        
                    tf.summary.scalar("Reward in each episode", total_reward, step=global_step)
                    tf.summary.scalar("Total step in each episode", step_in_ep, step=global_step)

                if FLAGS.eval_on_train and global_step % FLAGS.eval_step == 0:
                    self.test(global_step)
                    break

        self._predator_agent.save_nn(global_step)
        self._eval.summarize()

    def get_action(self, obs_n,weight, schedule_n, global_step, train=True):

        act_n = [0] * len(obs_n)
        self.epsilon = max(self.epsilon - epsilon_dec, epsilon_min)

        # Action of predator
          # with prob. epsilon
        predator_obs = [obs_n[i] for i in self._agent_profile['predator']['idx']]
        predator_action,probs = self._predator_agent.act(predator_obs,weight, schedule_n)

        for i, idx in enumerate(self._agent_profile['predator']['idx']):
            act_n[idx] = predator_action[i]

        # Action of prey
        for i, idx in enumerate(self._agent_profile['prey']['idx']):
            act_n[idx] = self._prey_agent[i].act(None)

        return np.array(act_n, dtype=np.int32),probs

    def get_schedule(self, obs_n, global_step, type, train=True):

        predator_obs = [obs_n[i] for i in self._agent_profile['predator']['idx']]

        if train and (global_step < FLAGS.m_size * FLAGS.pre_train_step or np.random.rand() < self.epsilon):
            # Exploration: Schedule k random agent
            priority = np.random.rand(self._n_predator)
            priority1 = np.random.rand(self._n_predator)
            sorted_agents = np.argsort(-priority1)
            allocation = np.zeros(4)
            remaining_bandwidth = FLAGS.capa
            softmax_weights = np.exp(priority1) / np.sum(np.exp(priority1))

    # Allocate bandwidth to agents starting from the largest priority1 values
            while remaining_bandwidth>0:
                for agent in sorted_agents:
                        agent_allocation = np.ceil((softmax_weights[agent]*remaining_bandwidth))
                        allocation[agent] = agent_allocation
                        remaining_bandwidth -= agent_allocation
            return allocation,priority,priority1
        else:
            # Exploitation
            return self._predator_agent.schedule(predator_obs,FLAGS.capa)

    def train_agents(self, state, obs_n, action_n,probs, reward_n, state_next, obs_n_next, schedule_n, priority,priority1, done,global_step):
        
        predator_obs = [obs_n[i] for i in self._agent_profile['predator']['idx']]
        predator_action = [action_n[i] for i in self._agent_profile['predator']['idx']]
        predator_reward = [reward_n[i] for i in self._agent_profile['predator']['idx']]
        predator_obs_next = [obs_n_next[i] for i in self._agent_profile['predator']['idx']]
        self._predator_agent.train(state, predator_obs, predator_action,probs, predator_reward,
                                   state_next, predator_obs_next, schedule_n, priority,priority1, done,global_step)
    #  The method concatenates the observations of the predator agents with the
    #  communication schedule history and returns the modified observation and state arrays.
    def get_h_obs_state(self, obs_n, state, h_schedule):
        obs_n_h = np.concatenate((obs_n[0:self._n_predator], h_schedule.reshape((self._n_predator,1))), axis=1)
        obs_final = list()
        for i in range(self._n_predator):
            obs_final.append(obs_n_h[i])
        for i in range(self._n_prey):
            obs_final.append(obs_n[self._n_predator + i])
        obs_n = np.array(obs_final)
        state = np.concatenate((state, h_schedule), axis=-1)

        return obs_n, state

    #  method and concatenates the observations of the predator
    #  agents with the updated communication schedule history to form the
    #  modified observation array. The method also concatenates the state of the environment 
    #  with the communication schedule history to form the modified state array.

    def get_obs_state_with_schedule(self, obs_n_ws, info_n, h_schedule_n, schedule_n=None, init=False):
        if not init:
            h_schedule_n = self.update_h_schedule(h_schedule_n, schedule_n)

        obs_n_h = np.concatenate((obs_n_ws[0:self._n_predator], h_schedule_n.reshape((self._n_predator,1))), axis=1)
        obs_final = list()
        for i in range(self._n_predator):
            obs_final.append(obs_n_h[i])
        for i in range(self._n_prey):
            obs_final.append(obs_n_ws[self._n_predator + i])
        # max_len = max([len(x) for x in obs_final])
        # padded_data = [np.pad(x, (0, max_len - len(x)), mode='constant') for x in obs_final]
        # obs_n = np.array(padded_data)
        obs_n = np.array(obs_final)
        state = np.concatenate((info_n[0]['state'], h_schedule_n), axis=-1)

        return obs_n, state, h_schedule_n
    #The update_h_schedule() method takes as input the current communication schedule history (h_schedule) 
    # and the communication schedule for the current step (schedule_n).
    def update_h_schedule(self, h_schedule, schedule_n):

        ret = h_schedule * 0.5 + schedule_n * 0.5
        return ret

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
            h_schedule_n = np.zeros(self._n_predator)
            obs_n, state, _ = self.get_obs_state_with_schedule(obs_n, info_n, h_schedule_n, init=True)

            while True:

                global_step += 1
                step_in_ep += 1

                schedule_n, priority,priority1 = self.get_schedule(obs_n, global_step, FLAGS.sched)
                action_n,probs = self.get_action(obs_n, priority1, schedule_n, global_step, False)
                obs_n_without_schedule, reward_n, done_n, info_n = self._env.step(action_n)
                obs_n_next, state_next, h_schedule_n = self.get_obs_state_with_schedule(obs_n_without_schedule, info_n, h_schedule_n, schedule_n)

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


