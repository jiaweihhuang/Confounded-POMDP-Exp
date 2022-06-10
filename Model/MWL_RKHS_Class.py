import tensorflow as tf
import numpy as np
import time
from functools import partial
from Model.Basic_Alg_Class import Basic_Alg

class MWL_RKHS(Basic_Alg):
    def __init__(self, obs_dim, act_dim, *, norm, q_net,
                       g_hidden_layers=[32, 32], g_lr=5e-3, 
                       scope='PO_MWL', med=10.0,
                       ep_len=1000, gamma=0.99,
                       other_configs={}, **kwargs):
        super().__init__(scope)

        assert act_dim == 2

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_list = [i for i in range(act_dim)]
        self.gamma = gamma
        self.ep_len = ep_len
        self.g_lr = g_lr
        self.g_hidden_layers = g_hidden_layers

        self.med_dist = med
        self.activation = tf.nn.relu

        self.q_net = q_net
        self.norm = norm

        self.other_configs = other_configs

        self.trainable_vars = []

        self.build_graph()
        self.build_loss_func()

        self.create_optimizer()
        self.build_estimation_graph()

        tf.get_default_session().run(
            [tf.variables_initializer(self.trainable_vars)]
        )
    
    def build_graph(self):     
        self.rew_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        self.init_obs_ph_1 = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim])
        self.next_obs_ph_1 = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim]) 

        self.obs_ph_1 = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim])
        self.acts_ph_1 = tf.placeholder(dtype=tf.int32, shape=[None, 1]) 
        self.acts_probs_ph_1 = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        self.done_ph_1 = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.is_init_ph_1 = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.factor_1 = tf.placeholder(dtype=tf.float32, shape=[None, 1])     # used for normalization

        self.init_obs_ph_2 = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim])
        self.next_obs_ph_2 = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim]) 

        self.obs_ph_2 = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim])
        self.acts_ph_2 = tf.placeholder(dtype=tf.int32, shape=[None, 1]) 
        self.acts_probs_ph_2 = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        self.done_ph_2 = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.is_init_ph_2 = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.factor_2 = tf.placeholder(dtype=tf.float32, shape=[None, 1])     # used for normalization

    def create_w_func(self, obs_tf, act_tf:tf.int32, reuse=False, last_activation=tf.abs):
        x = obs_tf

        with tf.variable_scope(self.scope + '_w_func', reuse=reuse):
            for h in self.g_hidden_layers:
                x = tf.layers.dense(x, h, activation=self.activation)
        
            out = tf.layers.dense(x, self.act_dim, activation=last_activation)
            mask = tf.one_hot(tf.squeeze(act_tf), depth=self.act_dim)
            out = tf.expand_dims(tf.reduce_sum(out * mask, axis=1), axis=1)
                
            return out

    def create_kernel_feature(self, obs, act):
        return tf.concat([obs * (1.0 - tf.cast(act, tf.float32)), obs * tf.cast(act, tf.float32)], axis=1)

    def build_loss_func(self):
        self.w1 = self.create_w_func(self.obs_ph_1, self.acts_ph_1, reuse=False)
        self.trainable_vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '_w_func')   
        self.w2 = self.create_w_func(self.obs_ph_2, self.acts_ph_2, reuse=True)
        
        epsilon = 1e-9
        normalizer_1 = tf.reduce_mean(tf.abs(self.w1) + epsilon)
        normalizer_2 = tf.reduce_mean(tf.abs(self.w2) + epsilon)

        self.w1 = self.w1 / normalizer_1
        self.w2 = self.w2 / normalizer_2
        self.normalizer_1 = normalizer_1

        # Compute Kernel Loss
        self.kernel_feature_1 = self.create_kernel_feature(self.obs_ph_1, self.acts_ph_1)
        self.kernel_feature_2 = self.create_kernel_feature(self.obs_ph_2, self.acts_ph_2)

        self.next_kernel_feature_list_1 = []
        self.next_kernel_feature_list_2 = []
        self.init_kernel_feature_list_1 = []
        self.init_kernel_feature_list_2 = []

        for act in self.act_list:
            self.next_kernel_feature_list_1.append(
                self.create_kernel_feature(self.next_obs_ph_1, act * tf.ones([tf.shape(self.next_obs_ph_1)[0], 1], dtype=tf.float32))
            )
            self.next_kernel_feature_list_2.append(
                self.create_kernel_feature(self.next_obs_ph_2, act * tf.ones([tf.shape(self.next_obs_ph_2)[0], 1], dtype=tf.float32))
            )
            self.init_kernel_feature_list_1.append(
                self.create_kernel_feature(self.init_obs_ph_1, act * tf.ones([tf.shape(self.init_obs_ph_1)[0], 1], dtype=tf.float32))
            )
            self.init_kernel_feature_list_2.append(
                self.create_kernel_feature(self.init_obs_ph_2, act * tf.ones([tf.shape(self.init_obs_ph_2)[0], 1], dtype=tf.float32))
            )

        if self.norm['type'] is not None:
            org_next_obs_1 = self.next_obs_ph_1 * self.norm['scale'] + self.norm['shift']
            org_next_obs_2 = self.next_obs_ph_2 * self.norm['scale'] + self.norm['shift']
            org_init_obs_1 = self.init_obs_ph_1 * self.norm['scale'] + self.norm['shift']
            org_init_obs_2 = self.init_obs_ph_2 * self.norm['scale'] + self.norm['shift']
        else:
            org_next_obs_1 = self.next_obs_ph_1
            org_next_obs_2 = self.next_obs_ph_2
            org_init_obs_1 = self.init_obs_ph_1
            org_init_obs_2 = self.init_obs_ph_2

        # shape = [None, 1]
        next_act_prob_list_1 = self.q_net.build_prob(org_next_obs_1, split=True)    
        next_act_prob_list_2 = self.q_net.build_prob(org_next_obs_2, split=True)
        init_act_prob_list_1 = self.q_net.build_prob(org_init_obs_1, split=True)
        init_act_prob_list_2 = self.q_net.build_prob(org_init_obs_2, split=True)

        kernel_feature_groups = []
        prob_mask = []
        coeff_lists = []
        w_func_group = []

        # Compute Term 1; 
        # K(O+, a'; \bar O+, \bar a')
        for next_act1 in self.act_list:
            for next_act2 in self.act_list:
                kernel_feature_groups.append(
                    (self.next_kernel_feature_list_1[next_act1], self.next_kernel_feature_list_2[next_act2])
                )
                prob_mask.append(
                    (next_act_prob_list_1[next_act1], tf.reshape(next_act_prob_list_2[next_act2], [1, -1]))
                )
                coeff_lists.append(
                    self.gamma ** 2
                )
                w_func_group.append(
                    (self.w1, self.w2)
                )

        # K(O+, a'; \bar O, \bar A)
        for next_act1 in self.act_list:
            kernel_feature_groups.append(
                (self.next_kernel_feature_list_1[next_act1], self.kernel_feature_2)
            )
            prob_mask.append(
                (next_act_prob_list_1[next_act1], tf.ones([1, tf.shape(self.kernel_feature_2)[0]]))
            )
            coeff_lists.append(
                -self.gamma
            )
            w_func_group.append(
                (self.w1, self.w2)
            )

        # K(O, A; \bar O+, \bar a')
        for next_act2 in self.act_list:
            kernel_feature_groups.append(
                (self.kernel_feature_1, self.next_kernel_feature_list_2[next_act2])
            )
            prob_mask.append(
                (tf.ones([tf.shape(self.kernel_feature_1)[0], 1]), tf.reshape(next_act_prob_list_2[next_act2], [1, -1]))
            )
            coeff_lists.append(
                -self.gamma
            )
            w_func_group.append(
                (self.w1, self.w2)
            )

        # K(O, A; \bar O, \bar A)
        kernel_feature_groups.append(
            (self.kernel_feature_1, self.kernel_feature_2)
        )
        prob_mask.append(
            None,
        )
        coeff_lists.append(
            1.0
        )
        w_func_group.append(
            (self.w1, self.w2)
        )

        # Compute Term 2
        pass    # There is no gradient

        # Compute Term 3
        w1_ones = tf.ones([tf.shape(self.init_obs_ph_1)[0], 1])
        w2_ones = tf.ones([tf.shape(self.init_obs_ph_2)[0], 1])

        # K(O^+, a'; \bar O_init, \bar a'_init)
        for next_act1 in self.act_list:
            for init_act2 in self.act_list:
                kernel_feature_groups.append(
                    (self.next_kernel_feature_list_1[next_act1], self.init_kernel_feature_list_2[init_act2])
                )
                prob_mask.append(
                    (next_act_prob_list_1[next_act1], tf.reshape(init_act_prob_list_2[init_act2], [1, -1]))
                )
                coeff_lists.append(
                    self.gamma * (1 - self.gamma)
                )
                w_func_group.append(
                    (self.w1, w2_ones)
                )

        # K(O_init, a'_init; \bar O^+, \bar a')
        for init_act1 in self.act_list:
            for next_act2 in self.act_list:
                kernel_feature_groups.append(
                    (self.init_kernel_feature_list_1[init_act1], self.next_kernel_feature_list_2[next_act2])
                )
                prob_mask.append(
                    (init_act_prob_list_1[init_act1], tf.reshape(next_act_prob_list_2[next_act2], [1, -1]))
                )
                coeff_lists.append(
                    self.gamma * (1 - self.gamma)
                )
                w_func_group.append(
                    (w1_ones, self.w2)
                )

        # K(O_init, a'_init; \bar O, \bar A)
        for init_act1 in self.act_list:
            kernel_feature_groups.append(
                (self.init_kernel_feature_list_1[init_act1], self.kernel_feature_2)
            )
            prob_mask.append(
                (init_act_prob_list_1[init_act1], tf.ones([1, tf.shape(self.kernel_feature_2)[0]]))
            )
            coeff_lists.append(
                -(1 - self.gamma)
            )
            w_func_group.append(
                (w1_ones, self.w2)
            )

        # K(O, A; \bar O_init, \bar a'_init)
        for init_act2 in self.act_list:
            kernel_feature_groups.append(
                (self.kernel_feature_1, self.init_kernel_feature_list_2[init_act2])
            )
            prob_mask.append(
                (tf.ones([tf.shape(self.kernel_feature_1)[0], 1]), tf.reshape(init_act_prob_list_2[init_act2], [1, -1]))
            )
            coeff_lists.append(
                -(1 - self.gamma)
            )
            w_func_group.append(
                (self.w1, w2_ones)
            )

        self.w_loss = 0.0

        for index in range(len(kernel_feature_groups)):
            kf1, kf2 = kernel_feature_groups[index]
            coeff = coeff_lists[index]
            w1, w2 = w_func_group[index]

            diff = tf.expand_dims(kf1, 1) - tf.expand_dims(kf2, 0)
            K = tf.exp(-tf.reduce_sum(tf.square(diff), axis=-1)/2.0/self.med_dist ** 2)
            
            if prob_mask[index] is not None:
                p1, p2 = prob_mask[index]
                p = p1 * p2
                K = K * p

            sample_num = tf.cast(tf.shape(K)[0] * tf.shape(K)[1], tf.float32)
            loss = coeff * tf.matmul(tf.matmul(tf.transpose(w1), K), w2) / sample_num

            self.w_loss += tf.squeeze(loss)

    def create_optimizer(self):
        self.w_opt = tf.train.AdamOptimizer(self.g_lr)
        self.w_train_op = self.w_opt.minimize(self.w_loss, var_list=self.get_all_vars_with_scope(self.scope + '_w_func'))
        self.trainable_vars += self.w_opt.variables()

    def build_estimation_graph(self):
        self.value_estimation = tf.reduce_mean(self.w1 * self.rew_ph) / (1.0 - self.gamma)

    def train_g(self, data):
        loss, _ = tf.get_default_session().run(
            [self.w_loss, self.w_train_op],
            feed_dict={
                self.init_obs_ph_1: data['init_obs_1'],
                self.obs_ph_1: data['obs_1'],
                self.next_obs_ph_1: data['next_obs_1'],     
                self.acts_ph_1: data['acts_1'],          
                self.acts_probs_ph_1: data['acts_probs_1'],
                self.factor_1: data['factor_1'],
                self.done_ph_1: data['done_1'],
                self.is_init_ph_1: data['is_init_1'],

                self.init_obs_ph_2: data['init_obs_2'],
                self.obs_ph_2: data['obs_2'],
                self.next_obs_ph_2: data['next_obs_2'],     
                self.acts_ph_2: data['acts_2'],          
                self.acts_probs_ph_2: data['acts_probs_2'],
                self.factor_2: data['factor_2'],
                self.done_ph_2: data['done_2'],
                self.is_init_ph_2: data['is_init_2'],

                self.q_net.tau_ph: self.q_net.default_tau,
            }
        )

        return loss

    def evaluation(self, dataset):        
        value = tf.get_default_session().run(
            [self.value_estimation],
            feed_dict={
                self.rew_ph: dataset['rews'],
                self.obs_ph_1: dataset['obs'],
                self.acts_ph_1: dataset['acts'],
                self.factor_1: dataset['factor'],
                self.is_init_ph_1: dataset['is_init'],
                self.q_net.tau_ph: self.q_net.default_tau,
            }
        )
        return value