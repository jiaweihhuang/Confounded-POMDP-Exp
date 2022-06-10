import numpy as np
import pickle
import argparse
from Env.OneDimProcess import Process, Sigmoid_Policy
import utils as U
import os

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', type = float, default = -1.0, help='w in sigmoid policy')
    parser.add_argument('-b', type = float, default = 1.0, help='b in sigmoid policy')
    
    parser.add_argument('--delta', type = float, default = 1.0, help='delta in dynamic')
    parser.add_argument('--inherent-noise', type = float, default = 0.5, help='inherent noise in transition')

    parser.add_argument('--obs-noise', type = float, default = 1.0, help='noise level for partial observation')

    parser.add_argument('--dataset-seed', type = int, nargs='+', default = [0], help='seed for dataset')
    parser.add_argument('--sample-size', type = int, default = 200000, help='inherent noise level')

    parser.add_argument('--gamma', type = float, default = 0.95, help='discounted factor')
    parser.add_argument('--ep-len', type = int, default = 100, help='episode length')

    args = parser.parse_args()

    return args


def main():
    args = get_parser()

    sample_size = args.sample_size
    max_ep_len = args.ep_len
    delta = args.delta
    obs_noise = args.obs_noise
    POMDP = args.obs_noise > 0.0

    parent_dir = './Dataset/OneDimProcess'
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    for dataset_seed in args.dataset_seed:
        U.set_seed(dataset_seed)

        env = Process(delta=delta, obs_noise=obs_noise, 
                POMDP=POMDP, max_ep_len=max_ep_len,
                inherent_noise=args.inherent_noise)

        policy = Sigmoid_Policy(w=args.w, b=args.b)

        dataset = generate_dataset(env, policy, sample_size, gamma=args.gamma)

        path = '{}/Process-{}-ep{}-delta{}-ObsNoise{}-InNoise{}-DatasetSeed{}-w{}b{}.pickle'.format(parent_dir, sample_size, max_ep_len, delta, obs_noise, args.inherent_noise, dataset_seed, args.w, args.b)
        with open(path, 'wb') as f:
            pickle.dump(dataset, f) 

def generate_dataset(env, policy, sample_size, gamma):
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
        # we use random noise as the last obs for initial state
        last_obs_list.append(np.random.normal()) 
        llast_obs_list.append(np.random.normal()) 
        init_last_obs_list.append(last_obs_list[-1])

        state = env.get_current_state()
        init_states_list.append(state)
        states_list.append(state)
        last_states_list.append(np.random.normal())
        llast_states_list.append(np.random.normal())

        is_init_list.append(True)

        done = False
        is_init_step = True
        
        factor = 1.0        

        while True:
            act = policy.sample_action(state)  # note that the behavior policy is based on state (instead of observation)
            
            next_obs, rew, done, _ = env.step(act)
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
                act = policy.sample_action(state)
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
            print('Total Sample till Now ', len(obs_list))

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
        'init_states': np.array(init_states_list)[:, np.newaxis],
        'states': np.array(states_list[:sample_size])[:, np.newaxis],
        'last_states': np.array(last_states_list[:sample_size])[:, np.newaxis],
        'llast_states': np.array(llast_states_list[:sample_size])[:, np.newaxis],
        'next_states': np.array(next_states_list[:sample_size])[:, np.newaxis],

        'init_last_obs': np.array(init_last_obs_list)[:, np.newaxis],
        'init_obs': np.array(init_obs_list)[:, np.newaxis],
        'last_obs': np.array(last_obs_list[:sample_size])[:, np.newaxis],
        'llast_obs': np.array(llast_obs_list[:sample_size])[:, np.newaxis],
        'obs': np.array(obs_list[:sample_size])[:, np.newaxis],
        'next_obs': np.array(next_obs_list[:sample_size])[:, np.newaxis],

        'init_acts': np.array(init_acts_list)[:, np.newaxis],
        'acts': np.array(act_list[:sample_size])[:, np.newaxis],
        'last_acts': np.array(last_act_list[:sample_size])[:, np.newaxis],
        'next_acts': np.array(next_acts_list[:sample_size])[:, np.newaxis],
        
        'rews': np.array(rew_list[:sample_size])[:, np.newaxis],
        'done': np.array(done_list[:sample_size])[:, np.newaxis],

        'is_init': np.array(is_init_list[:sample_size])[:, np.newaxis],
        'step_num': np.array(step_num_list[:sample_size])[:, np.newaxis],
    }


if __name__ == '__main__':
    main()