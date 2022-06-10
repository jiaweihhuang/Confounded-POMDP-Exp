from cmath import log
import numpy as np
import pickle
import os
import argparse
from Model.Linear_Feature_Model import Linear_Class, Feature_Dataset

class BinaryEnv():
    def __init__(self, obs_eps, transit_eps, ep_len=100):
        self.obs_eps = obs_eps
        self.transit_eps = transit_eps
        self.ep_len = ep_len

    def reset(self):
        self.counter = 0
        if np.random.rand() < 0.5:
            self.state = 0.0
        else:
            self.state = 1.0

        self.obs = self.create_obs()

        return self.obs

    # transition model P(S'=S) = 1-self.transit_eps
    def step(self, action):
        self.counter += 1
        
        reward = self.state * action
        if np.random.rand() > self.transit_eps:
            self.state = self.state
        else:
            self.state = 1.0 - self.state
        done = self.counter >= self.ep_len

        self.obs = self.create_obs()

        return self.obs, reward, done, {}

    # emission model P(O=S) = 1-self.obs_eps
    def create_obs(self):
        if np.random.rand() > self.obs_eps:
            obs = self.state
        else:
            obs = 1.0 - self.state
        return obs

    def get_obs(self):
        return self.obs

    def get_current_state(self):
        return self.state

class Policy():
    def __init__(self, epsilon):
        self.epsilon = epsilon

    # take action pi(A=O/S) w.p. 1-self.epsilon
    def get_action(self, obs):
        if np.random.rand() > self.epsilon:
            return obs
        else:
            return 1.0 - obs

    def get_prob_with_act(self, obs, act):
        return 1.0 * (obs == act) * (1 - 2 * self.epsilon) + self.epsilon

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obs-eps', type = float, default = 0.2, help='size of obs noise')
    parser.add_argument('--trans-eps', type = float, default = 0.2, help='size of obs noise')
    parser.add_argument('--ep-len', type = int, default = 100, help='size of obs noise')
    parser.add_argument('--ep-num', type = int, default = 2000, help='size of obs noise')
    parser.add_argument('--gamma', type = float, default = 0.95, help='size of obs noise')
    
    parser.add_argument('--target-eps', type = float, default = 0.5, help='size of obs noise')
    parser.add_argument('--behavior-eps', type = float, default = 0.2, help='size of obs noise')

    args = parser.parse_args()

    return args

def generate_data(env, ep_num, policy, gamma=0.95):
    sample_size = ep_num * env.ep_len

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
        last_obs_list.append(np.random.randint(2)) 
        llast_obs_list.append(np.random.randint(2)) 
        init_last_obs_list.append(last_obs_list[-1])

        state = env.get_current_state()
        init_states_list.append(state)
        states_list.append(state)
        last_states_list.append(np.random.randint(2))
        llast_states_list.append(np.random.randint(2))

        is_init_list.append(True)

        done = False
        is_init_step = True
        
        factor = 1.0        

        while True:
            act = policy.get_action(state) # here the behavior policy is based on state (instead of noisy observation)
            
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
                act = policy.get_action(state)
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

def eval_target_policy(env, policy, gamma, ep_num):
    ep_rets = []

    for i in range(ep_num):
        done = False
        ep_rets.append(0.0)

        obs = env.reset()
        factor = 1.0
        while not done:
            act = policy.get_action(obs)

            obs, rew, done, _ = env.step(act)

            ep_rets[-1] += rew * factor
            factor *= gamma
        
    return np.mean(ep_rets)


def main():
    args = get_parser()

    obs_dim = 1
    act_dim = 2
    feature_dim = 4
    ep_num = args.ep_num

    args.obs_eps = args.behavior_eps

    env = BinaryEnv(args.obs_eps, args.trans_eps, args.ep_len)
    
    log_dir = './Dataset/Binary/Tr{}_Obs{}_TargetPi{}_BehPi{}_EpNum{}.pickle'.format(args.trans_eps, args.obs_eps, args.target_eps, args.behavior_eps, ep_num)

    target_policy = Policy(epsilon=args.target_eps)
    behavior_policy = Policy(epsilon=args.behavior_eps)

    if not os.path.exists(log_dir):
        if not os.path.exists('./Dataset/Binary'):
            os.makedirs('./Dataset/Binary')
        dataset = generate_data(env, ep_num=ep_num, policy=behavior_policy, gamma=args.gamma)
        with open(log_dir, 'wb') as f:
            pickle.dump(dataset, f)
    else:
        with open(log_dir, 'rb') as f:
            dataset = pickle.load(f)

    # compute \pi_e(A|O) for all (O, A) in the dataset in advance
    dataset['acts_probs'] = target_policy.get_prob_with_act(dataset['obs'], dataset['acts'])
    dataset['last_acts_probs'] = target_policy.get_prob_with_act(dataset['last_obs'], dataset['last_acts'])

    dataset['init_acts_probs'] = [None] * act_dim
    dataset['next_acts_probs'] = [None] * act_dim
    for act in range(act_dim):
        dataset['init_acts_probs'][act] = target_policy.get_prob_with_act(dataset['init_obs'], act * np.ones(shape=[dataset['init_obs'].shape[0], 1]))
        dataset['next_acts_probs'][act] = target_policy.get_prob_with_act(dataset['next_obs'], act * np.ones(shape=[dataset['next_obs'].shape[0], 1]))

        print(dataset['next_acts_probs'][act].shape)

    # compute the true value:
    true_val = eval_target_policy(env, target_policy, gamma=args.gamma, ep_num=2000)
    print('true_val ', true_val)

    # learn
    feature_dataset = Feature_Dataset(dataset, num_acts=2, feature_dim=feature_dim,
            alpha=None, feature_type='one-hot',)
    linear_class = Linear_Class(
            obs_dim=obs_dim, 
            obs_ft_dim=feature_dim,
            num_acts=act_dim,
            gamma=args.gamma,
            dataset=dataset,
            feature_dataset=feature_dataset,
    )
    estimator, baseline_estimator = linear_class.learn()

    print('estimator ', estimator)
    print('baseline_estimator ', baseline_estimator)

if __name__ == '__main__':
    main()
