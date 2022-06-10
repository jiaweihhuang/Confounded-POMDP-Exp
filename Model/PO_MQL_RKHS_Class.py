import tensorflow as tf
from Model.Basic_Alg_Class import Basic_Alg

class PO_MQL_RKHS(Basic_Alg):
    def __init__(self, obs_dim, act_dim, *, norm, q_net,
                       g_hidden_layers=[256, 256], g_lr=5e-3, 
                       scope='PO_MQL', med=10.0,
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
        self.build_estimation_graph()

        tf.get_default_session().run(
            [tf.variables_initializer(self.trainable_vars)]
        )
    
    def build_graph(self):     
        self.rews_ph_1 = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.next_obs_ph_1 = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim]) 
        self.last_obs_ph_1 = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim]) 
        self.obs_ph_1 = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim])
        self.acts_ph_1 = tf.placeholder(dtype=tf.int32, shape=[None, 1])
        self.acts_probs_ph_1 = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.done_ph_1 = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        self.rews_ph_2 = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.next_obs_ph_2 = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim]) 
        self.last_obs_ph_2 = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim]) 
        self.obs_ph_2 = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim])
        self.acts_ph_2 = tf.placeholder(dtype=tf.int32, shape=[None, 1])
        self.acts_probs_ph_2 = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.done_ph_2 = tf.placeholder(dtype=tf.float32, shape=[None, 1])


    def create_g_func(self, obs_tf, act_tf:tf.int32, reuse=False):
        last_activation = None
        with tf.variable_scope(self.scope + '_g_func', reuse=reuse):
            x = obs_tf

            for h in self.g_hidden_layers:
                x = tf.layers.dense(x, h, activation=self.activation)

            out = tf.layers.dense(x, self.act_dim, activation=last_activation)
            mask = tf.one_hot(tf.squeeze(act_tf), depth=self.act_dim)
            out = tf.expand_dims(tf.reduce_sum(out * mask, axis=1), axis=1)

            return out

    def create_kernel_feature(self, obs, act):
        return tf.concat([obs * (1.0 - tf.cast(act, tf.float32)), obs * tf.cast(act, tf.float32)], axis=1)

    def build_loss_func(self, normalize=True):
        # compute g(A, O)
        self.g_AO_1 = self.create_g_func(self.obs_ph_1, self.acts_ph_1, reuse=False)
        self.trainable_vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '_g_func')    
        self.g_AO_2 = self.create_g_func(self.obs_ph_2, self.acts_ph_2, reuse=True)

        # compute R+\gamma \sum_{a'} g(a', O+)
        if self.norm['type'] is not None:
            org_next_obs_1 = self.next_obs_ph_1 * self.norm['scale'] + self.norm['shift']
            org_next_obs_2 = self.next_obs_ph_2 * self.norm['scale'] + self.norm['shift']
        else:
            org_next_obs_1 = self.next_obs_ph_1
            org_next_obs_2 = self.next_obs_ph_2
            
        next_acts_prob_list_1 = self.q_net.build_prob(org_next_obs_1, split=True)
        next_acts_prob_list_2 = self.q_net.build_prob(org_next_obs_2, split=True)

        self.sum_g_AO_plus_1 = 0.0
        self.sum_g_AO_plus_2 = 0.0
        for act in self.act_list:
            g_next_AO_1 = self.create_g_func(self.next_obs_ph_1, act * tf.ones([tf.shape(self.next_obs_ph_1)[0], 1], dtype=tf.int32), reuse=True)

            g_next_AO_2 = self.create_g_func(self.next_obs_ph_2, act * tf.ones([tf.shape(self.next_obs_ph_2)[0], 1], dtype=tf.int32), reuse=True)

            g_next_AO_1 = g_next_AO_1 * next_acts_prob_list_1[act]
            g_next_AO_2 = g_next_AO_2 * next_acts_prob_list_2[act]

            self.sum_g_AO_plus_1 += g_next_AO_1
            self.sum_g_AO_plus_2 += g_next_AO_2

        # compute \pi^e(A|O)
        self.err_1 = self.acts_probs_ph_1 * (self.rews_ph_1 + (1.0 - self.done_ph_1) * self.gamma * self.sum_g_AO_plus_1 - self.g_AO_1)
        self.err_2 = self.acts_probs_ph_2 * (self.rews_ph_2 + (1.0 - self.done_ph_2) * self.gamma * self.sum_g_AO_plus_2 - self.g_AO_2)
       
        kf1 = self.create_kernel_feature(self.last_obs_ph_1, tf.cast(self.acts_ph_1, tf.float32))
        kf2 = self.create_kernel_feature(self.last_obs_ph_2, tf.cast(self.acts_ph_2, tf.float32))

        diff = tf.expand_dims(kf1, 1) - tf.expand_dims(kf2, 0)
        K = tf.exp(-tf.reduce_sum(tf.square(diff), axis=-1)/2.0/self.med_dist ** 2)

        sample_num = tf.cast(tf.shape(K)[0] * tf.shape(K)[1], tf.float32)
        self.g_loss = tf.squeeze(tf.matmul(tf.matmul(tf.transpose(self.err_1), K), self.err_2)) / sample_num

        # Optimizer
        self.g_opt = tf.train.AdamOptimizer(self.g_lr)
        self.g_train_op = self.g_opt.minimize(self.g_loss, var_list=self.get_all_vars_with_scope(self.scope + '_g_func'))
        self.trainable_vars += self.g_opt.variables()


    def build_estimation_graph(self):
        self.init_obs_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim])

        value_estimation = 0.0
        if self.norm['type'] is not None:
            org_init_obs_ph = self.init_obs_ph * self.norm['scale'] + self.norm['shift']
        else:
            org_init_obs_ph = self.init_obs_ph
        init_acts_prob = self.q_net.build_prob(org_init_obs_ph)
        
        for act in self.act_list:
            value_est = self.create_g_func(self.init_obs_ph, act * tf.ones([tf.shape(self.init_obs_ph)[0], 1], dtype=tf.int32), reuse=True)
            value_est = value_est * init_acts_prob[act]
            value_estimation += value_est
        
        self.value_estimation = tf.reduce_mean(value_estimation)


    def train_g(self, data):
        loss, _ = tf.get_default_session().run(
            [self.g_loss, self.g_train_op],
            feed_dict={
                self.obs_ph_1: data['obs_1'],
                self.last_obs_ph_1: data['last_obs_1'],
                self.next_obs_ph_1: data['next_obs_1'],     
                self.acts_ph_1: data['acts_1'],          
                self.acts_probs_ph_1: data['acts_probs_1'],
                self.rews_ph_1: data['rews_1'],
                self.done_ph_1: data['done_1'],

                self.obs_ph_2: data['obs_2'],
                self.last_obs_ph_2: data['last_obs_2'],
                self.next_obs_ph_2: data['next_obs_2'],     
                self.acts_ph_2: data['acts_2'],          
                self.acts_probs_ph_2: data['acts_probs_2'],
                self.rews_ph_2: data['rews_2'],
                self.done_ph_2: data['done_2'],

                self.q_net.tau_ph: self.q_net.default_tau,
            }
        )
        return loss

    def evaluation(self, dataset):        
        value = tf.get_default_session().run(
            [self.value_estimation],
            feed_dict={
                self.init_obs_ph: dataset['init_obs'],
                self.q_net.tau_ph: self.q_net.default_tau,
            }
        )
        return value