import numpy as np
import pickle
import argparse
import sys

import utils as U
from Model.PO_MWL_RKHS_Class import PO_MWL_RKHS
from Model.PO_MQL_RKHS_Class import PO_MQL_RKHS
from Model.MQL_RKHS_Class import MQL_RKHS
from Model.MWL_RKHS_Class import MWL_RKHS
from Model.DR_Class import DR
from Model.PO_DR_Class import PO_DR
from Model.Q_Network_Class import Q_network
from Env.CartPole import CartPoleEnv
from Logger.logger_PO import ope_log_class
from multiprocessing import Pool
from copy import deepcopy
from functools import partial


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iter', type = int, default = 10000, help='training iteration')

    parser.add_argument('--gs', type = int, nargs='+', default = [256, 256], help='neural network size for q')
    parser.add_argument('--g-lr', type = float, default = 5e-3, help='learning rate for q')
    parser.add_argument('--bs', type = int, default = 500, help='batch size')  
    parser.add_argument('--gamma', type = float, default = 0.95, help='discounted factor')
    
    parser.add_argument('--norm', type = str, default = 'std_norm', choices=['std_norm'], help='normalization type')
    parser.add_argument('--ep-len', type = int, default = 1000, help='episode length')

    parser.add_argument('--dataset-seed', type = int, nargs='+', default = [100], help='random seed to generate dataset')
    parser.add_argument('--seed', type = int, nargs='+', default = [1000], help='random seed')
    parser.add_argument('--target-tau', type = float, nargs='+', default = [3.0], help='policy temperature for target policy')
    parser.add_argument('--sample-size', type = int, nargs='+', default = [200000], help='# trajectories in dataset')   
    
    parser.add_argument('--kernel-bw-tau', type = float, default = 1.0, help='kernel temperature for bw')
    parser.add_argument('--kernel-bv-tau', type = float, default = 0.1, help='kernel temperature for bv')

    parser.add_argument('--POMDP', action='store_true', default=False, help='whether use partial observation')
    parser.add_argument('--PO-type', type=str, default='noise', choices=['noise', 'mask'], help='how to create observation')
    parser.add_argument('--obs-noise', type=float, nargs='+', default=0.1)

    parser.add_argument('--baseline', action='store_true', help='whether to use baseline (MQL)')

    parser.add_argument('--behavior-tau', type = float, default = 5.0, help='behavior policy temperature')

    parser.add_argument('--mask-index', type=int, nargs='+', default=[0])
    args = parser.parse_args()

    return args


def sample_data_double(dataset, sample_num):
    data_size = dataset['obs'].shape[0]

    index_1 = np.random.choice(data_size, sample_num)
    init_index_1 = np.random.choice(dataset['init_obs'].shape[0], 200)

    index_2 = np.random.choice(data_size, sample_num)
    init_index_2 = np.random.choice(dataset['init_obs'].shape[0], 200)

    return {
        'obs_1': dataset['obs'][index_1],
        'last_obs_1': dataset['last_obs'][index_1],
        'next_obs_1': dataset['next_obs'][index_1],
        'acts_1': dataset['acts'][index_1],
        'last_acts_1': dataset['last_acts'][index_1],
        'acts_probs_1': dataset['acts_probs'][index_1],
        'last_acts_probs_1': dataset['last_acts_probs'][index_1],
        'factor_1': dataset['factor'][index_1],
        'done_1': dataset['done'][index_1],
        'is_init_1': dataset['is_init'][index_1],
        'init_obs_1': dataset['init_obs'][init_index_1],
        'rews_1': dataset['rews'][index_1],

        'obs_2': dataset['obs'][index_2],
        'last_obs_2': dataset['last_obs'][index_2],
        'next_obs_2': dataset['next_obs'][index_2],
        'acts_2': dataset['acts'][index_2],
        'last_acts_2': dataset['last_acts'][index_2],
        'acts_probs_2': dataset['acts_probs'][index_2],
        'last_acts_probs_2': dataset['last_acts_probs'][index_2],
        'factor_2': dataset['factor'][index_2],
        'done_2': dataset['done'][index_2],
        'is_init_2': dataset['is_init'][index_2],
        'init_obs_2': dataset['init_obs'][init_index_2],
        'rews_2': dataset['rews'][index_2],
    }


def estimate_med_dist(dataset):
    data = sample_data_double(dataset, 5000)
    obs_acts_1 = np.concatenate([data['obs_1'] * (1.0 - data['acts_1']), data['obs_1'] * data['acts_1']], axis=1)
    obs_acts_2 = np.concatenate([data['obs_2'] * (1.0 - data['acts_2']), data['obs_2'] * data['acts_2']], axis=1)

    obs_act_m0 = np.median(np.sqrt(np.sum(np.square(obs_acts_1[None, :, :] - obs_acts_2[:, None, :]), axis=-1)))

    last_obs_acts_1 = np.concatenate([data['last_obs_1'] * (1.0 - data['acts_1']), data['last_obs_1'] * data['acts_1']], axis=1)
    last_obs_acts_2 = np.concatenate([data['last_obs_2'] * (1.0 - data['acts_2']), data['last_obs_2'] * data['acts_2']], axis=1)

    last_obs_act_m0 = np.median(np.sqrt(np.sum(np.square(last_obs_acts_1[None, :, :] - last_obs_acts_2[:, None, :]), axis=-1)))

    print('median value is ', last_obs_act_m0)
    return obs_act_m0, last_obs_act_m0


def main(args):
    command = sys.executable + " " + " ".join(sys.argv)
    env_name = "CartPole"
    ep_len = args.ep_len
    dataset_seed = args.dataset_seed
    seed = args.seed

    U.set_seed(dataset_seed + seed)

    env = CartPoleEnv(max_ep_len=ep_len,
            seed=seed, 
            partial_obs=args.POMDP, 
            partial_obs_type=args.PO_type,
            mask_index=args.mask_index,
            obs_noise=args.obs_noise)

    obs_dim = 4
    act_dim = 2

    sess = U.make_session()
    sess.__enter__()

    '''load evaluation policy'''
    q_net = Q_network(obs_dim, act_dim, seed=100, default_tau=args.target_tau)
    U.initialize_all_vars()

    # we evaluate softened obs-based policy

    if args.PO_type == 'noise':
        q_net.load_model('./CartPole_Model/PO_Model_Noise_0.1_Expert/Model')
        file_name = './Dataset/{}/CartPole-ep1000-tau{}-ObsNoise{}-DatasetSeed{}.pickle'.format(args.sample_size, args.behavior_tau, args.obs_noise, args.dataset_seed)
    elif args.PO_type == 'mask':
        q_net.load_model('./CartPole_Model/PO_Model_Mask_Expert/Model')
        file_name = './Dataset/{}/CartPole-ep1000-tau{}-MaskIndex{}-DatasetSeed{}.pickle'.format(args.sample_size, args.behavior_tau, args.mask_index, args.dataset_seed)
    else:
        raise NotImplementedError


    with open(file_name, 'rb') as f:
        dataset = pickle.load(f)

    norm_type = args.norm
    gamma = args.gamma
    if 'factor' not in dataset.keys():
        dataset['factor'] = np.array([gamma ** (i % 1000) for i in range(dataset['obs'].shape[0])]).reshape([-1, 1])

    dataset['done'] = dataset['done'] * 1.0
    dataset['is_init'] = dataset['is_init'] * 1.0

    # compute \pi_e(A|O) for all (O, A) in the dataset in advance
    dataset['acts_probs'] = q_net.get_prob_with_act(dataset['obs'], dataset['acts'], q_net.default_tau)
    dataset['last_acts_probs'] = q_net.get_prob_with_act(dataset['last_obs'], dataset['last_acts'], q_net.default_tau)

    # normalize observation
    if norm_type == 'std_norm':
        obs_mean = np.mean(dataset['obs'], axis=0, keepdims=True)
        obs_std = np.std(dataset['obs'], axis=0, keepdims=True) + 1e-9

        dataset['obs'] = (dataset['obs'] - obs_mean) / obs_std
        dataset['next_obs'] = (dataset['next_obs'] - obs_mean) / obs_std
        dataset['init_obs'] = (dataset['init_obs'] - obs_mean) / obs_std

        norm = {'type': norm_type, 'shift': obs_mean, 'scale': obs_std}
    else:
        norm = {'type': None, 'shift': None, 'scale': None}            

    obs_act_m0, last_obs_act_m0 = estimate_med_dist(dataset)

    # if evaluating baseline methods, then use MQL/MWL and DR
    # otherwise, use PO_MQL/PO_MWL and DR
    if args.baseline:
        PO_MQL_Alg_Class = partial(MQL_RKHS, med=obs_act_m0 * args.kernel_bv_tau)
        PO_MWL_Alg_Class = partial(MWL_RKHS, med=obs_act_m0 * args.kernel_bw_tau)
        PO_DR_Alg_Class = DR
    else:
        PO_MQL_Alg_Class = partial(PO_MQL_RKHS, med=last_obs_act_m0 * args.kernel_bv_tau)
        PO_MWL_Alg_Class = partial(PO_MWL_RKHS, med=obs_act_m0 * args.kernel_bw_tau)
        PO_DR_Alg_Class = PO_DR

    PO_MWL_alg = PO_MWL_Alg_Class(obs_dim, act_dim, q_net=q_net,
            g_hidden_layers=args.gs, g_lr=args.g_lr, 
            gamma=args.gamma, norm=norm, scope='PO_MWL',
            other_configs=None)
    PO_MQL_alg = PO_MQL_Alg_Class(obs_dim, act_dim, q_net=q_net,
            g_hidden_layers=args.gs, g_lr=args.g_lr, 
            ep_len=args.ep_len, gamma=args.gamma, norm=norm, scope='PO_MQL',
            other_configs=None)
    DR_alg = PO_DR_Alg_Class(obs_dim, act_dim, 
            PO_MQL_alg=PO_MQL_alg, PO_MWL_alg=PO_MWL_alg,
            q_net=q_net, norm=norm, gamma=args.gamma)

    sess.graph.finalize()

    value_true, _ = U.eval_policy_cartpole(env, q_net, ep_num=10, gamma=args.gamma, prt=True)

    log_name = 'log_seed{}_bv{}_bw{}_Tau{}.pickle'.format(args.seed, args.kernel_bv_tau, args.kernel_bw_tau, args.target_tau)
    exp_info = 'ObsNoise{}_Size{}'.format(args.obs_noise, args.sample_size)

    if args.baseline:
        logger = ope_log_class(path='./log/{}/Est_DR_Baseline/Dataset{}'.format(exp_info, args.dataset_seed), name=log_name, tau=args.target_tau, env_name=env_name, value_true=value_true)
    else:
        logger = ope_log_class(path='./log/{}/Est_DR/Dataset{}'.format(exp_info, args.dataset_seed), name=log_name, tau=args.target_tau, env_name=env_name, value_true=value_true)

    prt_interval = 500
    log_interval = 100

    for Iter in range(args.iter):  
        data = sample_data_double(dataset, args.bs)
        bw_loss = PO_MWL_alg.train_g(data)

        data = sample_data_double(dataset, args.bs)
        bv_loss = PO_MQL_alg.train_g(data)

        if Iter % prt_interval == 0:
            print('-------------------------------------')
            print('Iter: {},\tbw loss {},\tbv loss {}'.format(Iter, bw_loss, bv_loss))

            print('True value: {}'.format(value_true))
            estimation(PO_MWL_alg, PO_MQL_alg, DR_alg, dataset, prt=True)
            print('-------------------------------------\n\n')

        if Iter % log_interval == 0:
            logger.update_info(Iter, {'bv_loss': bv_loss, 'bw_loss': bw_loss})

            PO_MWL_est_value, PO_MQL_est_value, DR_est_value = estimation(PO_MWL_alg, PO_MQL_alg, DR_alg, dataset)
            logger.update_info(Iter, {'(PO-)MWL': PO_MWL_est_value, '(PO-)MQL': PO_MQL_est_value, '(PO-)DR': DR_est_value})

            logger.dump()

def estimation(PO_MWL_alg, PO_MQL_alg, DR_alg, dataset, prt=False):
    PO_MWL_est_value = PO_MWL_alg.evaluation(dataset)
    PO_MQL_est_value = PO_MQL_alg.evaluation(dataset)
    DR_est_value = DR_alg.evaluation(dataset)

    if prt:
        print('\nPO MWL Est Value {}'.format(PO_MWL_est_value))
        print('\nPO MQL Est Value {}'.format(PO_MQL_est_value))
        print('\nDR Est Value {}'.format(DR_est_value))
        
    return PO_MWL_est_value, PO_MQL_est_value, DR_est_value
        

if __name__ == '__main__':
    args = get_parser()

    args_list = []
    for dataset_seed in args.dataset_seed:
        for target_tau in args.target_tau:
            for seed in args.seed:
                for sample_size in args.sample_size:
                    for obs_noise in args.obs_noise:
                        args_copy = deepcopy(args)
                        args_copy.dataset_seed = dataset_seed
                        args_copy.seed = seed
                        args_copy.target_tau = target_tau
                        args_copy.sample_size = sample_size
                        args_copy.obs_noise = obs_noise
                        args_list.append(args_copy)

    with Pool(processes=len(args_list), maxtasksperchild=1) as p:
        p.map(main, args_list, chunksize=1)