import tensorflow as tf
import numpy as np
import time

class Q_network():
    def __init__(self, obs_dim, act_dim, *, 
                    seed, 
                    default_tau=1,
                    hidden_layers=[64],
                    name="deepq/q_func",
                    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        assert self.act_dim == 2

        self.name = name
        self.seed = seed
        self.hidden_layers = hidden_layers
        self.default_tau = default_tau
        
        self.build_network()

    def set_default_tau(self, tau):
        self.default_tau = tau

    def build_network(self, reuse=False):
        with tf.variable_scope(self.name, reuse):
            self.obs_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim], name='obs')
            x = self.obs_ph
            index = 0
            for h in self.hidden_layers:
                x = tf.contrib.layers.fully_connected(x, h, activation_fn=tf.nn.relu)
                index += 1

            self.q_value = tf.contrib.layers.fully_connected(x, self.act_dim, activation_fn=None)
            
            # temperature
            self.tau_ph = tf.placeholder(dtype=tf.float32, shape=[])
            self.logits = self.q_value / self.tau_ph

            # probability
            self.prob = tf.nn.softmax(self.logits, axis=1)

            # actions
            self.greedy_action = tf.argmax(self.q_value, axis=1)
            random_action = tf.squeeze(tf.multinomial(self.logits, 1, seed=self.seed))
            self.random_action = tf.reshape(random_action, tf.shape(self.greedy_action))

            mask = tf.one_hot(tf.squeeze(self.random_action), depth=self.act_dim)

            self.act_prob = tf.expand_dims(tf.reduce_sum(self.prob * mask, axis=1), axis=1)
        
            ''' if given act'''
            self.act_index_ph = tf.placeholder(dtype=tf.int32, shape=[None, 1])
            mask_2 = tf.one_hot(tf.squeeze(self.act_index_ph), depth=self.act_dim)
            self.act_prob_given = tf.reduce_sum(self.prob * mask_2, axis=1, keep_dims=True)
        
    def build_random_policy(self, obs_ph, reuse=True):
        with tf.variable_scope(self.name, reuse=reuse):
            x = obs_ph
            index = 0
            for h in self.hidden_layers:
                x = tf.contrib.layers.fully_connected(x, h, activation_fn=tf.nn.relu)
                index += 1

            q_value = tf.contrib.layers.fully_connected(x, self.act_dim, activation_fn=None)
            
            # temperature
            logits = q_value / self.tau_ph

            self.seed += 10
            random_action = tf.multinomial(logits, 1, seed=self.seed)

            return random_action

    def build_random_policy_and_ratio(self, obs_ph, reuse=True, b_tau=1.5):
        with tf.variable_scope(self.name, reuse=reuse):
            x = obs_ph
            index = 0
            for h in self.hidden_layers:
                x = tf.contrib.layers.fully_connected(x, h, activation_fn=tf.nn.relu)
                index += 1

            q_value = tf.contrib.layers.fully_connected(x, self.act_dim, activation_fn=None)
            
            # temperature
            b_logits = q_value / b_tau
            t_logits = q_value / self.tau_ph

            b_prob = tf.nn.softmax(b_logits, axis=1)
            t_prob = tf.nn.softmax(t_logits, axis=1)

            ratio_all = t_prob / b_prob

            self.seed += 10
            random_action = tf.multinomial(t_logits, 1, seed=self.seed)

            mask = tf.one_hot(tf.squeeze(random_action), depth=2)
            ratio = tf.reduce_sum(ratio_all * mask, axis=1, keep_dims=True)

            return random_action, ratio

    def build_prob(self, obs_ph, reuse=True, split=True):
        assert reuse is True
        with tf.variable_scope(self.name, reuse=reuse):
            x = obs_ph
            index = 0
            for h in self.hidden_layers:
                x = tf.contrib.layers.fully_connected(x, h, activation_fn=tf.nn.relu)
                index += 1

            q_value = tf.contrib.layers.fully_connected(x, self.act_dim, activation_fn=None)
            
            # temperature
            logits = q_value / self.tau_ph

            # probability
            prob = tf.stop_gradient(tf.nn.softmax(logits, axis=1))

            if split:
                return tf.split(prob, self.act_dim, axis=1)
            else:
                return prob


    def get_act_and_prob(self, obs):
        return tf.get_default_session().run(
            [self.random_action, self.act_prob],
            feed_dict={
                self.obs_ph: obs,
                self.tau_ph: self.default_tau,
            }
        )
    
    def get_prob_with_act(self, obs, act, tau):
        return tf.get_default_session().run(
            self.act_prob_given,
            feed_dict={
                self.obs_ph: obs,
                self.act_index_ph: act,
                self.tau_ph: tau,
            }
        )

    # TODO, merge sample_action and sample_action
    def sample_action(self, obs, norm={'type': 'None'}):
        if norm['type'] != 'None':
            org_obs = obs * norm['scale'] + norm['shift']
        else:
            org_obs = obs
        return tf.get_default_session().run(
            self.random_action, 
            feed_dict={
                self.obs_ph: org_obs,
                self.tau_ph: self.default_tau,
            }
        )


    def load_model(self, path):       
        saver = tf.train.Saver()
        saver.restore(tf.get_default_session(), path)

    @property
    def all_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    
    def get_q_value(self, obs):
        return tf.get_default_session().run(
            self.q_value,
            feed_dict={
                self.obs_ph: obs,
            }
        )

    def get_logits(self, obs):
        return tf.get_default_session().run(
            self.logits,
            feed_dict={
                self.obs_ph: obs,
            }
        )

    def get_probabilities(self, obs, norm={'type': 'None'}):
        if norm['type'] != 'None':
            org_obs = obs * norm['scale'] + norm['shift']
        else:
            org_obs = obs
        return tf.get_default_session().run(
            self.prob,
            feed_dict={
                self.obs_ph: org_obs,
                self.tau_ph: self.default_tau,
            }
        )