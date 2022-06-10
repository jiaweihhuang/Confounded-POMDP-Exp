import numpy as np
import pickle
import argparse
import sys
import os

import utils as U
from Env.OneDimProcess import Process, Sigmoid_Policy
from Model.Linear_Feature_Model import Feature_Dataset, Linear_Class


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iter', type = int, default = 10000, help='training iteration')

    parser.add_argument('--gamma', type = float, default = 0.95, help='discounted factor')
    parser.add_argument('--ep-len', type = int, default = 100, help='episode length')

    parser.add_argument('--dataset-seed', type = int, nargs='+', default = [0], help='random seed of dataset')
    parser.add_argument('--seed', type = int, nargs='+', default = [1000], help='random seed for RBF feature')
    parser.add_argument('--sample-size', type = int, default = 200000, help='# trajectories in dataset')   
    
    parser.add_argument('--POMDP', action='store_true', default=False, help='whether use partial observation')
    parser.add_argument('--feature-dim', type=int, default=100)
    
    parser.add_argument('--alpha', type = float, default = 5.0, help='kernel temperature for RBF feature')
    
    parser.add_argument('-w', type = float, nargs='+', default = -1.0, help='kernel temperature for bw')
    parser.add_argument('-b', type = float, default = 1.0, help='kernel temperature for bv')
    parser.add_argument('--delta', type=float, default=1.0)
    
    parser.add_argument('--b-w', type = float, default = -1.0, help='w of behavior policy')
    parser.add_argument('--b-b', type = float, default = 1.0, help='b of behavior policy')

    parser.add_argument('--inherent-noise', type = float, default = 0.5, help='inherent noise level')
    parser.add_argument('--obs-noise', type=float, default=0.5)

    args = parser.parse_args()

    return args


def main(args):
    command = sys.executable + " " + " ".join(sys.argv)
    dataset_seed = args.dataset_seed
    seed = args.seed

    obs_dim = 1
    act_dim = 2

    sess = U.make_session()
    sess.__enter__()

    # Evaluate one dataset
    sample_size = args.sample_size
    max_ep_len = args.ep_len
    delta = args.delta
    obs_noise = args.obs_noise

    log_path = './log/Toy_Linear_Est_DR/FDim{}_ObsNoise{}_InNoise{}_Alpha{}_S{}.pickle'.format(args.feature_dim, obs_noise, args.inherent_noise, args.alpha, args.sample_size)

    if not os.path.exists('./log/Toy_Linear_Est_DR'):
        os.makedirs('./log/Toy_Linear_Est_DR')

    log = {}

    # for each dataset 
    for dataset_seed in args.dataset_seed:
        file_name = './Dataset/OneDimProcess/Process-{}-ep{}-delta{}-ObsNoise{}-InNoise{}-DatasetSeed{}-w{}b{}.pickle'.format(sample_size, max_ep_len, delta, obs_noise, args.inherent_noise, dataset_seed, args.b_w, args.b_b)
        print('Evaluating with Dataset ', file_name)

        with open(file_name, 'rb') as f:
            dataset = pickle.load(f)

        gamma = args.gamma
        if 'factor' not in dataset.keys():
            dataset['factor'] = np.array([gamma ** (i % 1000) for i in range(dataset['obs'].shape[0])]).reshape([-1, 1])

        dataset['done'] = dataset['done'] * 1.0
        dataset['is_init'] = dataset['is_init'] * 1.0

        log[dataset_seed] = {}

        # for each choice of w
        for w in args.w:
            print('Evaluating for w={}'.format(w))

            target_policy = Sigmoid_Policy(w=w, b=args.b)

            # compute \pi_e(A|O) for all (O, A) in the dataset in advance
            dataset['acts_probs'] = target_policy.get_prob_with_act(dataset['obs'], dataset['acts'])
            dataset['last_acts_probs'] = target_policy.get_prob_with_act(dataset['last_obs'], dataset['last_acts'])

            dataset['init_acts_probs'] = [None] * act_dim
            dataset['next_acts_probs'] = [None] * act_dim
            for act in range(act_dim):
                dataset['init_acts_probs'][act] = target_policy.get_prob_with_act(dataset['init_obs'], act * np.ones(shape=[dataset['init_obs'].shape[0], 1]))
                dataset['next_acts_probs'][act] = target_policy.get_prob_with_act(dataset['next_obs'], act * np.ones(shape=[dataset['next_obs'].shape[0], 1]))

            log[dataset_seed][w] = {}

            # for each random seed to generate the RBF feature
            for seed in args.seed:
                print('Evaluating for seed={}'.format(seed))
                # create feature dataset
                feature_dataset = Feature_Dataset(
                                dataset, 
                                num_acts=act_dim, 
                                feature_dim=args.feature_dim,
                                alpha=args.alpha,
                                seed=seed,
                            )

                # create linear estimator
                linear_estimator = Linear_Class(
                                    obs_dim=obs_dim, 
                                    obs_ft_dim=args.feature_dim,
                                    num_acts=act_dim,
                                    gamma=args.gamma,
                                    dataset=dataset,
                                    feature_dataset=feature_dataset,
                                )

                PO_estimator, Baseline_estimator = linear_estimator.learn()

                log[dataset_seed][w][seed] = {
                    'PO_estimator': PO_estimator,
                    'Baseline_estimator': Baseline_estimator,
                }

                print(log[dataset_seed][w][seed])

        with open(log_path, 'wb') as f:
            pickle.dump(log, f)


if __name__ == '__main__':
    args = get_parser()
    main(args)