

import numpy as np
import tensorflow as tf
import config
from agents.schednet import comm


FLAGS = config.flags.FLAGS

#Hyperparameters: 

gamma = 0.9  # reward discount factor
# Hidden layers:
h_critic = FLAGS.h_critic
h1_critic = h_critic  # hidden layer 1 size for the critic
h2_critic = h_critic  # hidden layer 2 size for the critic
h3_critic = h_critic  # hidden layer 3 size for the critic

# Learning rates: 
lr_actor = 0.00001   # learning rate for the actor
lr_critic = 0.0001  # learning rate for the critic
lr_decay = 1  # learning rate decay (per episode)

# The soft target update rate. 
tau = 5e-2 
# It controls the rate at which the target network is updated towards the main network during training.
# A smaller value results in a slower update rate.


class ObservationSelectNetwork:

    def __init__(self, sess, n_agent, obs_dim_per_unit, nn_id=None):

        self.sess = sess # The TensorFlow session.
        self.n_agent = n_agent
        self.obs_dim_per_unit = obs_dim_per_unit
        self.action_dim=obs_dim_per_unit
        if nn_id == None: # An optional identifier to distinguish different instances of the actor network.
            scope = 'obs_selector'
        else:
            scope = 'obs_selector_' + str(nn_id)

        # placeholders
        self.state_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, obs_dim_per_unit*n_agent])
        self.next_state_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, obs_dim_per_unit*n_agent])
        self.count_ph=tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, 1])
        # Concat action space
        self.action_ph = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, n_agent])
        self.schedule_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.n_agent * FLAGS.capa])

        # It is reshaped using tf.reshape and transformed into a one-hot encoding using tf.one_hot.
        # It has a shape of [-1, action_dim * n_agent], where -1 implies that the first dimension is automatically inferred
        # based on the batch size.
        self.a_onehot = tf.reshape(tf.one_hot(self.action_ph, self.action_dim, 1.0, 0.0), [-1, obs_dim_per_unit * n_agent])
        self.td_errors = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 1])

        # indicators (go into target computation). a boolean scalar indicating whether the network is in training mode (True) or not (False).
        self.is_training_ph = tf.compat.v1.placeholder(dtype=tf.bool, shape=())


        #  Action selector
        #  This line calls the generate_actor_network method to generate the actions using the actor network.
        #  The state_ph and schedule_ph are passed as inputs to the method. The trainable parameter is set to True, indicating that the weights of the actor network should be updated during training.
        #  The resulting actions are stored in the self.actions attribute.
        with tf.compat.v1.variable_scope(scope):
            self.actions = self.generate_obs_selector(self.state_ph, trainable=True)

        # Actor loss function (mean Q-values under current policy with regularization)
        self.actor_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        self.responsible = tf.multiply(self.actions, self.a_onehot)
        log_prob = tf.compat.v1.log(tf.reduce_sum(self.responsible, axis=1, keepdims=True))
        entropy = -tf.reduce_sum(self.actions*tf.math.log(self.actions), 1) # calculates the entropy of the actions 
        #  Computes the actor loss by multiplying the log probabilities (log_prob) with the TD errors (self.td_errors),
        #  Adding a regularization term of 0.01 * entropy, and summing them all together.
        self.loss = tf.reduce_sum(-(tf.multiply(log_prob, self.td_errors) + 0.01*entropy)) 

        # This calculates the gradients of the actor loss (self.loss) with respect to the actor variables (self.actor_vars)
        var_grads = tf.gradients(self.loss, self.actor_vars)
        self.actor_train_op = tf.compat.v1.train.AdamOptimizer(lr_actor * lr_decay).apply_gradients(zip(var_grads,self.actor_vars))
    # method prepares the observations for each agent, passes them to
    # comm.generate_comm_network, and returns the output of the actor network

    def generate_obs_selector(self, obs, trainable, share=False):

        obs_list = list()
        for i in range(self.n_agent):
            obs_list.append(obs[:, i * self.obs_dim_per_unit:(i + 1) * self.obs_dim_per_unit])

        ret = obs.generate_obs_network(obs_list, self.obs_dim_per_unit, self.action_dim, self.n_agent)
        return ret

    def action_for_state(self, state_ph):

        return self.sess.run(self.actions,
                             feed_dict={self.state_ph: state_ph,
                                        self.is_training_ph: False})

    def training_actor(self, state_ph, action_ph, td_errors):
   
        return self.sess.run(self.actor_train_op,
                             feed_dict={self.state_ph: state_ph,
                                        self.action_ph: action_ph,
                                        self.td_errors: td_errors,
                                        self.is_training_ph: True})
  



class ObservationCriticNetwork:
    def __init__(self, sess, n_agent, state_dim, nn_id=None):

        self.sess = sess
        self.n_agent = n_agent
        self.state_dim = 28

        if nn_id == None:
            scope = 'critic'
        else:
            scope = 'critic_' + str(nn_id)

        # placeholders
        self.state_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.state_dim])
        self.reward_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None])
        self.next_state_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.state_dim])
        
        self.is_not_terminal_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None])  # indicators (go into target computation)
        self.is_training_ph = tf.compat.v1.placeholder(dtype=tf.bool, shape=())  # for dropout
    
        
        with tf.compat.v1.variable_scope(scope):
            self.q_values = self.generate_critic_network(self.state_ph, trainable=True)
        with tf.compat.v1.variable_scope('slow_target_'+scope):
            slow_q_values = self.generate_critic_network(self.next_state_ph, trainable=False)

            self.slow_q_values = tf.stop_gradient(slow_q_values[0])
  

        # One step TD targets y_i for (s,a) from experience replay
        # = r_i + gamma*Q_slow(s',mu_slow(s')) if s' is not terminal
        # = r_i if s' terminal
        targets = tf.expand_dims(self.reward_ph, 1) + tf.expand_dims(self.is_not_terminal_ph, 1) * gamma * self.slow_q_values


        # 1-step temporal difference errors
        self.td_errors = targets - self.q_values

       # compute critic gradients
        optimizer = tf.compat.v1.train.AdamOptimizer(lr_critic * lr_decay)
        critic_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        critic_loss = tf.reduce_mean(tf.square(self.td_errors))
        

# minimize critic loss
        self.critic_train_op = tf.compat.v1.train.AdamOptimizer(lr_critic * lr_decay).minimize(critic_loss, var_list=critic_vars)
        slow_target_critic_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='slow_target_'+scope)
        update_slow_target_ops_c = []
        for i, slow_target_var in enumerate(slow_target_critic_vars):
            update_slow_target_critic_op = slow_target_var.assign(tau * critic_vars[i] + (1 - tau) * slow_target_var)
            update_slow_target_ops_c.append(update_slow_target_critic_op)
        self.update_slow_targets_op_c = tf.group(*update_slow_target_ops_c)

        self.scheduler_gradients = tf.gradients(self.sch_q_values, self.priority_ph)[0]
        self.scheduler_gradients1 = tf.gradients(self.sch_q_values1, self.priority_ph1)[0]
  


    def generate_critic_network(self, s, trainable):
        state_action = s

        hidden = tf.keras.layers.Dense(h1_critic, activation=tf.nn.relu,
                                 kernel_initializer=tf.random_normal_initializer(0., .1),
                                 bias_initializer=tf.constant_initializer(0.1),  
                                 use_bias=True, trainable=trainable, name='dense_c1')(state_action)

        hidden_2 = tf.keras.layers.Dense(h2_critic, activation=tf.nn.relu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),
                                   bias_initializer=tf.constant_initializer(0.1),  
                                   use_bias=True, trainable=trainable, name='dense_c2')(hidden)

        hidden_3 = tf.keras.layers.Dense( h3_critic, activation=tf.nn.relu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),
                                   bias_initializer=tf.constant_initializer(0.1), 
                                   use_bias=True, trainable=trainable, name='dense_c3')(hidden_2)

        q_values = tf.keras.layers.Dense(1, trainable=trainable,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),
                                   bias_initializer=tf.constant_initializer(0.1),  
                                   name='dense_c4', use_bias=False)(hidden_3)
        
       
        return q_values

    def training_critic(self, state_ph, reward_ph, next_state_ph, is_not_terminal_ph):

        return self.sess.run([self.td_errors, self.critic_train_op],
                             feed_dict={self.state_ph: state_ph,
                                        self.reward_ph: reward_ph,
                                        self.next_state_ph: next_state_ph,
                                        self.is_not_terminal_ph: is_not_terminal_ph,
                                        self.is_training_ph: True})

    def training_target_critic(self):
        return self.sess.run(self.update_slow_targets_op_c,
                             feed_dict={self.is_training_ph: False})

    
