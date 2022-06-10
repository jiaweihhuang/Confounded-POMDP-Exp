import tensorflow as tf
from Model.Basic_Alg_Class import Basic_Alg

class PO_DR(Basic_Alg):
    def __init__(self, obs_dim, act_dim, *, norm, q_net,
                       PO_MQL_alg, PO_MWL_alg,
                       scope='DR', 
                       gamma=0.99,
                       other_configs={}):
        super().__init__(scope)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.q_net = q_net
        self.norm = norm
        self.act_list = [i for i in range(act_dim)]

        self.PO_MQL_alg = PO_MQL_alg
        self.PO_MWL_alg = PO_MWL_alg
        self.gamma = gamma

        self.build_graph()

    def build_graph(self):
        self.rew_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.next_obs_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim]) 
        self.init_obs_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim]) 
        self.last_obs_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim]) 
        self.obs_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.obs_dim])
        self.acts_ph = tf.placeholder(dtype=tf.int32, shape=[None, 1])
        self.acts_probs_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.done_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        # compute bw function
        self.bw_A_Om = self.PO_MWL_alg.create_g_func(self.last_obs_ph, self.acts_ph, reuse=True)
        epsilon = 1e-9
        self.bw_normalizer = tf.reduce_mean(self.acts_probs_ph * self.bw_A_Om) + epsilon
        self.bw_A_Om_normalized = self.bw_A_Om / self.bw_normalizer

        # compute init estimation
        self.init_bv_estimation = 0.0
        # Note that we use reparameterization form to learn bv
        if self.norm['type'] is not None:
            org_init_obs_ph = self.init_obs_ph * self.norm['scale'] + self.norm['shift']
        else:
            org_init_obs_ph = self.init_obs_ph
        init_acts_prob = self.q_net.build_prob(org_init_obs_ph)
        for act in self.act_list:
            self.init_bv_estimation += self.PO_MQL_alg.create_g_func(self.init_obs_ph, act * tf.ones([tf.shape(self.init_obs_ph)[0], 1], dtype=tf.int32), reuse=True) * init_acts_prob[act]
        self.init_bv_estimation = tf.reduce_mean(self.init_bv_estimation)
        
        # compute delta
        self.bv_AO = self.PO_MQL_alg.create_g_func(self.obs_ph, self.acts_ph, reuse=True)
        self.sum_bv_AO_plus = 0.0
        
        if self.norm['type'] is not None:
            org_next_obs_ph = self.next_obs_ph * self.norm['scale'] + self.norm['shift']
        else:
            org_next_obs_ph = self.next_obs_ph
        next_acts_prob = self.q_net.build_prob(org_next_obs_ph)  

        for act in self.act_list:
            self.sum_bv_AO_plus += self.PO_MQL_alg.create_g_func(self.next_obs_ph, act * tf.ones([tf.shape(self.next_obs_ph)[0], 1], dtype=tf.int32), reuse=True) * next_acts_prob[act]
            
        self.delta = self.acts_probs_ph * (self.rew_ph + self.gamma * (1.0 - self.done_ph) * self.sum_bv_AO_plus - self.bv_AO)
            
        # compute dr estimator
        self.dr_estimator = self.init_bv_estimation + tf.reduce_mean(self.bw_A_Om_normalized * self.delta) / (1.0 - self.gamma)


    def evaluation(self, dataset):    
        value = tf.get_default_session().run(
            self.dr_estimator,
            feed_dict={
                self.rew_ph: dataset['rews'],
                self.acts_probs_ph: dataset['acts_probs'],
                self.obs_ph: dataset['obs'],
                self.next_obs_ph: dataset['next_obs'],
                self.last_obs_ph: dataset['last_obs'],
                self.init_obs_ph: dataset['init_obs'],
                self.acts_ph: dataset['acts'],
                self.done_ph: dataset['done'],
                self.q_net.tau_ph: self.q_net.default_tau,
            }
        )
        return value