import numpy as np
import tensorflow as tf
import time

def make_session(num_cpu=4):
    """Returns a session that will use <num_cpu> CPU's only"""
    gpu_options = tf.GPUOptions(allow_growth=True)
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=num_cpu,
        gpu_options=gpu_options)
    return tf.Session(config=tf_config)


def initialize_all_vars():
    tf.get_default_session().run(tf.global_variables_initializer())


def set_seed(seed):
    tf.set_random_seed(seed + 1)
    np.random.seed(seed + 2)


def eval_policy_cartpole(env, alg, ep_num=10, gamma=None, prt=False, save_data=False, POMDP=False):
    rew_list = []
    obs_list = []
    act_list = []

    ep_rews = []

    undiscounted = []

    assert hasattr(alg, 'sample_action')
    for i in range(ep_num):
        if prt:
            if i > 0:
                print('Traj {} {}'.format(i, ep_rews[-1]))
        obs = env.reset()
        done = False
        factor = 1.0

        ep_rew = 0.0

        undiscounted.append(0.0)

        while not done:
            act = np.squeeze(alg.sample_action([obs]))

            obs_list.append(obs)
            act_list.append(act)

            obs, rew, done, _ = env.step(act)
            
            undiscounted[-1] += rew

            rew *= factor
            factor *= gamma

            ep_rew += rew
            rew_list.append(rew)

        ep_rews.append(ep_rew)

    print(np.mean(undiscounted))
    if save_data:
        return np.mean(ep_rews), np.array(obs_list), np.array(act_list).reshape([-1, 1])
    else:
        return np.mean(ep_rews), ep_rews


def get_percentile(data):
    ptr = []
    for i in range(10):
        ptr.append(np.percentile(data, i * 10 + 5))
    print(ptr)