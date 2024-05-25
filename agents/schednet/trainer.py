from __future__ import print_function, division, absolute_import
import math 
import itertools
import numpy as np
from agents.schednet.agent import PredatorAgent
from agents.simple_agent import RandomAgent
from agents.evaluation import Evaluation
import logging
import config
from envs.gui import canvas
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import random
import Environment_marl

FLAGS = config.flags.FLAGS
writer = SummaryWriter("log")


def convert_observation(observation):
    state = [item for sublist in observation for item in sublist]
    return state
    # We have a 4 by 33 array.
    # Each observation shape is 33 and hence the state dimension is 132


# ################## SETTINGS ######################
up_lanes = [i/2.0 for i in [3.5/2,3.5/2 + 3.5,250+3.5/2, 250+3.5+3.5/2, 500+3.5/2, 500+3.5+3.5/2]]
down_lanes = [i/2.0 for i in [250-3.5-3.5/2,250-3.5/2,500-3.5-3.5/2,500-3.5/2,750-3.5-3.5/2,750-3.5/2]]
left_lanes = [i/2.0 for i in [3.5/2,3.5/2 + 3.5,433+3.5/2, 433+3.5+3.5/2, 866+3.5/2, 866+3.5+3.5/2]]
right_lanes = [i/2.0 for i in [433-3.5-3.5/2,433-3.5/2,866-3.5-3.5/2,866-3.5/2,1299-3.5-3.5/2,1299-3.5/2]]

width = 750/2
height = 1298/2



label = 'marl_model'

n_veh = 4
n_neighbor = 1
n_RB = n_veh

env1 = Environment_marl.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, n_neighbor)
env1.new_random_game()  # initialize parameters in env

n_episode = 3000
n_step_per_episode = int(env1.time_slow/env1.time_fast)
V2I_rate_list = []
V2V_success_list=[]
epsi_anneal_length = int(0.95*n_episode)

def convert_observation(observation):
    state = [item for sublist in observation for item in sublist]
    return state
    # We have a 4 by 33 array.
    # Each observation shape is 33 and hence the state dimension is 132
def convert_action(action):
    output_list = [[[x %4, x //4 ]] for x in action]
    return output_list
def get_state(env, idx=(0,0), ind_episode=1., epsi=0.02):
    """ Get state from the environment """
    done=False
    # V2I_channel = (env.V2I_channels_with_fastfading[idx[0], :] - 80) / 60
    V2I_fast = (env.V2I_channels_with_fastfading[idx[0], :] - env.V2I_channels_abs[idx[0]] + 10)/35
    # V2V_channel = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :] - 80) / 60
    V2V_fast = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :] - env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] + 10)/35

    # V2V_interference has 4 values one for each V2V channel
    V2V_interference = (-env.V2V_Interference_all[idx[0], idx[1], :] - 60) / 60
    # The channel gain of its own transmitter to base station
    V2I_abs = (env.V2I_channels_abs[idx[0]] - 80) / 60.0
    # The channel gain of own signal gk[m] for all m 
    V2V_abs = (env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] - 80)/60.0
    
    # Remaining V2V payload 
    load_remaining = np.asarray([env.demand[idx[0], idx[1]] / env.demand_size])
    # Remaining V2V time 
    time_remaining = np.asarray([env.individual_time_limit[idx[0], idx[1]] / env.time_slow])
    if load_remaining==0:
        done=True

    # return np.concatenate((np.reshape(V2V_channel, -1), V2V_interference, V2I_abs, V2V_abs, time_remaining, load_remaining, np.asarray([ind_episode, epsi])))
    return np.concatenate((V2I_fast, np.reshape(V2V_fast, -1), V2V_interference, np.asarray([V2I_abs]), V2V_abs, time_remaining, load_remaining, np.asarray([ind_episode, epsi]))),done



class Trainer(object):
  
    def __init__(self, env):
        self.epsilon_decay=.0000025
        self.min_epsilon = 0.001
        self.env = env
        self._eval = Evaluation()
        
        # State and obs additionally include history information
        self._state_dim = 132

        self._obs_dim = obs_dim=33
        
        # Predator agent
        self._predator_agent = PredatorAgent(n_agent=4,
                                             action_dim=16,
                                             state_dim=self._state_dim,
                                             obs_dim=self._obs_dim)
        

        self.epsilon = .5 # Init value for epsilon

        
    # Main function of the algorithm. 
    def generate_action_space(self):
        values = [0, 1, 2, 3,4]
        action_space = []

        # Generate all possible combinations of values
        combinations = list(itertools.product(values, repeat=4))

        # Filter combinations where the sum of each element is not above 3
        valid_combinations = [combo for combo in combinations if sum(combo)==1]

        # Convert combinations to action vectors
        for combo in valid_combinations:
            action_space.append(list(combo))
        return action_space
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
    
    def learn(self):
        action_space=self.generate_action_space()
        global_step = 0
        episode_num = 0
        print_flag = True
        episode_count=0
        for i_episode in range(n_episode):  
            V2I_rate_per_episode = []
            V2V_success_per_episode=[]
            total_rate=0
            # env.new_random_game()
            done= [False for i in range(4)] 
            ep_reward=0
            episode_count=0
            # env.new_random_game()
            print("-------------------------")
            print('Episode:', i_episode)
            
            if i_episode%50 == 0:
                self.env.renew_positions() # update vehicle position
                self.env.renew_neighbor()
                self.env.renew_channel() # update channel slow fading
                self.env.renew_channels_fastfading() # update channel fast fading
            self.env.demand = self.env.demand_size * np.ones((n_veh, n_neighbor))
            self.env.individual_time_limit = self.env.time_slow * np.ones((n_veh, n_neighbor))
            self.env.active_links = np.ones((n_veh, n_neighbor), dtype='bool')
            done=False
            stepinep=0
            while stepinep<=n_step_per_episode:
                self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay)
                global_step+=1
                state_old_all = []
                state_new_all=[]
                done_all=[]
                # First dimenstion is for n_veh, second dimenstion is for n_neighbors and third is for 0 or 1 (Binary)
                for i in range(n_veh):
                    for j in range(n_neighbor):
                        state,done = get_state(self.env, [i, j], i_episode/(n_episode-1), self.epsilon)
                        state_old_all.append(state)
                        done_all.append(done)
                state=convert_observation(state_old_all)
                schedule_n,action, priority = self.get_schedule(state, global_step,action_space, FLAGS.sched)
                action_n,count,messages = self.get_action(state, schedule_n, global_step)
                action_all_training=convert_action(action_n)
                action_temp = action_all_training.copy()
                action_temp=np.array(action_temp)
                train_reward,V2I_Rate,V2V_Rate,V2V_success= self.env.act_for_training(action_temp)
                V2I_rate_per_episode.append(np.sum(V2I_Rate))
                V2V_success_per_episode.append(np.sum(V2V_success))
                ep_reward+=train_reward         
                if global_step%1000==0:
                    print(priority)
                    print(action_space[action])
                    print(messages)   
                    print(self.epsilon)
                # print(record_reward)
                self.env.renew_channels_fastfading()
                self.env.Compute_Interference(action_temp)
                for i in range(n_veh):
                    for j in range(n_neighbor):
                        state_new,done1 = get_state(self.env, [i, j], i_episode/(n_episode-1), self.epsilon)
                        state_new_all.append(state_new)
                state_next=convert_observation(state_new_all)  
                
                episode_count+=count[0][0]
                done_single=np.sum(np.array(done_all))
                stepinep+=1
                self.train_agents(state,action, state, action_n, train_reward, state_next, state_next, schedule_n,count, priority, done_single,global_step)

            writer.add_scalar("Total reward in each episode", ep_reward, global_step=global_step)  # assuming each value is a scalar
            writer.add_scalar("V2I rate in each episode", np.mean(V2I_rate_per_episode), global_step=global_step)  # assuming each value is a scalar
            writer.add_scalar("V2V rate in each episode", np.mean(V2V_success_per_episode), global_step=global_step)  # assuming each value is a scalar
            episode_num += 1
        V2I_rate_list.append(np.mean(V2I_rate_per_episode))
        print(V2I_rate_list)

                # done_single = sum(done_n) > 0
            
                    # if is_episode_done(done_n, global_step):

                        # if FLAGS.gui:
                        #     self.canvas.draw(state_next * FLAGS.map_size, [0]*self._n_predator, "Train", True)
                        # print(episode_count)
                        # done = True
                        
                                    
        
                    
                    # tf.summary.scalar("Total messages in each episode", episode_count, step=global_step)
                
                # if FLAGS.eval_on_train and global_step % FLAGS.eval_step == 0:
                #     self.test(global_step)
                #     break
                
                     
        # self._predator_agent.save_nn(global_step)
        # self._eval.summarize()
            

    def get_action(self, obs_n, schedule_n, global_step, train=True):

        act_n = [0] * len(obs_n)
        

        # Action of predator
        # predator_obs = [obs_n[i] for i in self._agent_profile['predator']['idx']]
        predator_action,count,messages = self._predator_agent.act(obs_n, schedule_n)

        return np.array(predator_action, dtype=np.int32),count,messages

    def get_schedule(self, obs_n, global_step,action_space, type, train=True):

        # predator_obs = [obs_n[i] for i in self._agent_profile['predator']['idx']]
        
        return self._predator_agent.schedule(obs_n,FLAGS.capa,self.epsilon,action_space)

    def train_agents(self, state,action, obs_n, action_n, reward_n, state_next, obs_n_next, schedule_n,count, priority, done,global_step):
        
        # predator_obs = [obs_n[i] for i in self._agent_profile['predator']['idx']]
        # predator_action = [action_n[i] for i in self._agent_profile['predator']['idx']]
        # predator_reward = [reward_n[i] for i in self._agent_profile['predator']['idx']]
        # predator_obs_next = [obs_n_next[i] for i in self._agent_profile['predator']['idx']]
        self._predator_agent.train(state,action, obs_n, action_n, reward_n,
                                   state_next, obs_n_next, schedule_n,count, priority, done,global_step)
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

        obs_n = np.array(obs_final,dtype=object)
        state = np.concatenate((info_n[0]['state'], h_schedule_n), axis=-1)

        return obs_n, state, h_schedule_n
    #The update_h_schedule() method takes as input the current communication schedule history (h_schedule) 
    # and the communication schedule for the current step (schedule_n).
    def update_h_schedule(self, h_schedule, schedule_n):
        schedule_n=np.array(schedule_n)
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
        action_space=self.generate_action_space()
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
                schedule_n,action, priority,results = self.get_schedule(obs_n, global_step,action_space, FLAGS.sched)
                action_n,count = self.get_action(obs_n, schedule_n, global_step, False)
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


