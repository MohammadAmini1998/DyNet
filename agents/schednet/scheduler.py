
import numpy as np
import tensorflow as tf
import config

FLAGS = config.flags.FLAGS

#Hyperparameters: 

gamma = 0.9  # reward discount factor
gammaComm=.99
# Hidden layers:
h_critic = FLAGS.h_critic
h1_critic = h_critic  # hidden layer 1 size for the critic
h2_critic = h_critic  # hidden layer 2 size for the critic
h3_critic = h_critic  # hidden layer 3 size for the critic

# Learning rates: 
lr_actor = 0.00001   # learning rate for the actor
lr_critic = 0.00001  # learning rate for the critic
lr_decay = 1  # learning rate decay (per episode)

# The soft target update rate. 
tau = 5e-2 
# It controls the rate at which the target network is updated towards the main network during training.
# A smaller value results in a slower update rate.

class Scheduler:
    def __init__(self, sess, n_agent, state_dim, nn_id=None):

        self.sess = sess
        self.n_agent = n_agent
        self.state_dim = 8
        self.action_dim=35
        
        if nn_id == None:
            scope = 'critic1'
        else:
            scope = 'critic_1' + str(nn_id)

        # placeholders
        self.state_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 8])
        self.reward_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None])
        self.next_state_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 8])
        self.action_ph=tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, 1])
        self.is_not_terminal_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None])  # indicators (go into target computation)
        self.is_training_ph = tf.compat.v1.placeholder(dtype=tf.bool, shape=())  # for dropout
        # The placeholders self.priority_ph and self.next_priority_ph are used to hold the priority
        # values associated with the experiences in prioritized experience replay.
        # In reinforcement learning, prioritized experience replay is a technique that
        # assigns priorities to experiences based on their TD errors. Experiences with higher
        # TD errors are considered more important and are sampled more frequently for training the critic network.
        # Second weight generator 
        with tf.compat.v1.variable_scope(scope):
            # Critic applied to state_ph
            self.q_values = self.generate_critic_network(self.state_ph,self.action_dim, trainable=True)

        # slow target critic network
        # slow_q_values[0] represents the Q-values predicted by the slow target critic network, while slow_q_values[1]
        # represents the scheduled Q-values predicted by the slow target network.
        with tf.compat.v1.variable_scope('slow_target_1'+scope):
            slow_q_values = self.generate_critic_network(self.next_state_ph,self.action_dim, trainable=False)

            #  This line applies the tf.stop_gradient
            #  function to the Q-values obtained from the slow
            #  target critic network. It prevents the gradients
            #  from flowing throughthese Q-values during backpropagation,
            #  effectively treating them as constant values. These Q-values are stored in the attribute self.slow_q_values.
            self.slow_q_values = tf.stop_gradient(slow_q_values[0])
            self.max_q_value = tf.reduce_max(self.slow_q_values)
          

        # One step TD targets y_i for (s,a) from experience replay
        # = r_i + gamma*Q_slow(s',mu_slow(s')) if s' is not terminal
        # = r_i if s' terminal
        self.selected_q_values = tf.gather(self.q_values, self.action_ph, axis=1)

        targets = tf.expand_dims(self.reward_ph, 1) + tf.expand_dims(self.is_not_terminal_ph, 1) * .90 * self.max_q_value


        # 1-step temporal difference errors
        self.td_errors = targets - self.selected_q_values

       # compute critic gradients
        optimizer = tf.compat.v1.train.AdamOptimizer(lr_critic * lr_decay)
        critic_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        critic_loss = tf.reduce_mean(tf.square(self.td_errors))
        

# minimize critic loss
        self.critic_train_op = tf.compat.v1.train.AdamOptimizer(lr_critic * lr_decay).minimize(critic_loss, var_list=critic_vars)
        slow_target_critic_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='slow_target_1'+scope)
        update_slow_target_ops_c = []
        for i, slow_target_var in enumerate(slow_target_critic_vars):
            update_slow_target_critic_op = slow_target_var.assign(tau * critic_vars[i] + (1 - tau) * slow_target_var)
            update_slow_target_ops_c.append(update_slow_target_critic_op)
        self.update_slow_targets_op_c = tf.group(*update_slow_target_ops_c)


    def generate_critic_network(self, s,action_dim, trainable):
        state_action = s
        
        hidden = tf.keras.layers.Dense(h1_critic, activation=tf.nn.relu,
                                 kernel_initializer=tf.random_normal_initializer(0., .1),
                                 bias_initializer=tf.constant_initializer(0.1),  
                                 use_bias=True, trainable=trainable, name='dense_c11')(state_action)

        hidden_2 = tf.keras.layers.Dense(h2_critic, activation=tf.nn.relu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),
                                   bias_initializer=tf.constant_initializer(0.1),  
                                   use_bias=True, trainable=trainable, name='dense_c21')(hidden)

        hidden_3 = tf.keras.layers.Dense( h3_critic, activation=tf.nn.relu,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),
                                   bias_initializer=tf.constant_initializer(0.1), 
                                   use_bias=True, trainable=trainable, name='dense_c31')(hidden_2)

        q_values = tf.keras.layers.Dense(action_dim, trainable=trainable,
                                   kernel_initializer=tf.random_normal_initializer(0., .1),
                                   bias_initializer=tf.constant_initializer(0.1),  
                                   name='dense_c41', use_bias=False)(hidden_3)
        
    
        return q_values

    def training_critic(self,action_ph, state_ph, reward_ph, next_state_ph, is_not_terminal_ph):

        return self.sess.run([self.td_errors, self.critic_train_op],
                             feed_dict={self.action_ph:action_ph,
                                        self.state_ph: state_ph,
                                        self.reward_ph: reward_ph,
                                        self.next_state_ph: next_state_ph,
                                        self.is_not_terminal_ph: is_not_terminal_ph,
                                        self.is_training_ph: True})

    def training_target_critic(self):
        return self.sess.run(self.update_slow_targets_op_c,
                             feed_dict={self.is_training_ph: False})
    
    def get_com_action(self,state_ph):
        return self.sess.run(self.q_values,
                             feed_dict={self.state_ph: state_ph,
                                        self.is_training_ph: False})


    