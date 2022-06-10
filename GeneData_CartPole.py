import numpy as np
import os
import pickle
import argparse
import utils as U
import tensorflow as tf

from Model.Q_Network_Class import Q_network
from Env.CartPole import CartPoleEnv

def generate_dataset(env, q_net=None, sample_size=200000, act_choices=[0,1], gamma=0.95):

    init_states_list = []
    last_states_list = []
    llast_states_list = []  # the state before last state
    states_list = []
    next_states_list = []

    init_obs_list = []
    init_last_obs_list = []
    last_obs_list = []
    llast_obs_list = []  # the obs before last obs
    obs_list = []
    next_obs_list = []

    init_acts_list = []
    act_list = []
    last_act_list = []
    next_acts_list = []

    rew_list = []
    done_list = []

    is_init_list = []
    step_num_list = []

    ep_num = 0
    total_return = 0.0
    while True:
        ep_num += 1
        step_num = 0

        obs = env.reset()
        init_obs_list.append(obs) 
        obs_list.append(obs)
        last_obs_list.append(np.random.normal(size=obs.shape)) 
        llast_obs_list.append(np.random.normal(size=obs.shape)) 
        init_last_obs_list.append(last_obs_list[-1])

        state = env.get_current_state()
        init_states_list.append(state)
        states_list.append(state)
        last_states_list.append(np.random.normal(size=state.shape))
        llast_states_list.append(np.random.normal(size=state.shape))

        is_init_list.append(True)

        done = False
        is_init_step = True
        
        factor = 1.0        

        while True:
            if q_net is None:
                act = np.random.choice(act_choices)
            else:
                act = q_net.sample_action([state])  # here the behavior policy is based on state (instead of noisy observation)
                act = act[0]
            
            next_obs, rew, done, _ = env.step([act_choices[act]])
            next_state = env.get_current_state()

            if is_init_step:
                is_init_step = False
                init_acts_list.append(act)
                last_act_list.append(np.zeros_like(act))
            else:
                next_acts_list.append(act)
                assert len(next_acts_list) == len(act_list), '{}!={}'.format(len(next_acts_list), len(act_list))

            act_list.append(act)
            rew_list.append(rew)
            next_obs_list.append(next_obs)
            next_states_list.append(next_state)
            done_list.append(done)
            step_num_list.append(step_num)
            
            step_num += 1
            factor *= gamma
            total_return += factor * rew

            if done:
                if q_net is None:
                    act = np.random.choice(act_choices)
                else:
                    act = q_net.sample_action([next_obs])[0]
                next_acts_list.append(act)
                break

            last_act_list.append(act)

            last_obs = obs
            last_state = state
            obs = next_obs
            state = next_state

            last_obs_list.append(last_obs)
            llast_obs_list.append(last_obs_list[-1])

            last_states_list.append(last_state)
            llast_states_list.append(last_states_list[-1])
            
            obs_list.append(obs)
            states_list.append(state)
            is_init_list.append(False)

        if ep_num % 100 == 0:
            print('\n\n')
            print('Average Discounted Return ', total_return / ep_num)
            print('Average Return ', np.sum(rew_list) / ep_num)
            print('Average Ep_Len ', len(rew_list) / ep_num)
            print('Sample Size till Now ', len(obs_list))

        if len(obs_list) >= sample_size:
            break

    print('Total samples:', len(obs_list))
    '''
    Return Shape:
    obs: [None, obs_dim],
    acts: [None, 1],
    next_obs: [None, obs_dim],
    rews: [None, 1]
    '''
    return {
        'init_states': np.array(init_states_list),
        'states': np.array(states_list[:sample_size]),
        'last_states': np.array(last_states_list[:sample_size]),
        'llast_states': np.array(llast_states_list[:sample_size]),
        'next_states': np.array(next_states_list[:sample_size]),

        'init_last_obs': np.array(init_last_obs_list),
        'init_obs': np.array(init_obs_list),
        'last_obs': np.array(last_obs_list[:sample_size]),
        'llast_obs': np.array(llast_obs_list[:sample_size]),
        'obs': np.array(obs_list[:sample_size]),
        'next_obs': np.array(next_obs_list[:sample_size]),

        'init_acts': np.array(init_acts_list)[:, np.newaxis],
        'acts': np.array(act_list[:sample_size])[:, np.newaxis],
        'last_acts': np.array(last_act_list[:sample_size])[:, np.newaxis],
        'next_acts': np.array(next_acts_list[:sample_size])[:, np.newaxis],
        
        'rews': np.array(rew_list[:sample_size])[:, np.newaxis],
        'done': np.array(done_list[:sample_size])[:, np.newaxis],

        'is_init': np.array(is_init_list[:sample_size])[:, np.newaxis],
        'step_num': np.array(step_num_list[:sample_size])[:, np.newaxis],
    }

def main():
    parser = argparse.ArgumentParser(description='Generate Dataset')
    parser.add_argument('--behavior-tau', type = float, default = 1.0, help='temperature of behavior policy')
    parser.add_argument('--dataset-seed', type = int, nargs='+', default = [100], help='seed')
    parser.add_argument('--sample-size', type = int, default = 200000, help='number of rollouts')
    parser.add_argument('--ep-len', type = int, default = 1000, help='episode length')
    
    parser.add_argument('--POMDP', action='store_true', default=False, help='whether use partial observation')
    parser.add_argument('--obs-noise', type=float, default=0.1)

    parser.add_argument('--PO-type', type=str, default='noise', choices=['noise', 'mask'], help='how to create observation')
    parser.add_argument('--mask-index', type=int, nargs='+', default=[0])

    args = parser.parse_args()

    env_name = "CartPole"

    ep_len = args.ep_len
    behavior_tau = args.behavior_tau
    sample_size = args.sample_size
    st_dim = 4
    act_dim = 2

    obs_dim = st_dim

    sess = U.make_session()
    sess.__enter__()

    assert behavior_tau > 0, 'behavior_tau should be positive value'    
    q_net = Q_network(st_dim, act_dim, seed=100, default_tau=behavior_tau)

    sess.run(tf.global_variables_initializer())

    q_net.load_model('./CartPole_Model/Full_Observation_Expert/Model')

    if not os.path.exists('./Dataset/{}'.format(sample_size)):
        os.makedirs('./Dataset/{}'.format(sample_size))

    for dataset_seed in args.dataset_seed:
        U.set_seed(dataset_seed)
        env = CartPoleEnv(max_ep_len=ep_len,
                seed=dataset_seed + 100,
                partial_obs=True,
                partial_obs_type=args.PO_type,
                mask_index=args.mask_index,
                obs_noise=args.obs_noise)
        dataset = generate_dataset(env, q_net, sample_size=sample_size)

        if args.PO_type == 'noise':
            path = './Dataset/{}/CartPole-ep{}-tau{}-ObsNoise{}-DatasetSeed{}.pickle'.format(sample_size, ep_len, behavior_tau, args.obs_noise, dataset_seed)
        elif args.PO_type == 'mask':
            path = './Dataset/{}/CartPole-ep{}-tau{}-MaskIndex{}-DatasetSeed{}.pickle'.format(sample_size, ep_len, behavior_tau, args.mask_index, dataset_seed)
        else:
            raise NotImplementedError

        with open(path, 'wb') as f:
            pickle.dump(dataset, f) 

if __name__ == '__main__':
    main()
