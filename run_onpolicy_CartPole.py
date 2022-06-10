import numpy as np
import argparse
import sys
import os
sys.path.append('..')

import utils as U
from Model.Q_Network_Class import Q_network
from Env.CartPole import CartPoleEnv
from Logger.logger import ope_log_class


def get_parser():
    parser = argparse.ArgumentParser(description='On Policy')
    parser.add_argument('--seed', type = int, default = 1000, help='random seed')
    parser.add_argument('--gamma', type = float, default = 0.99, help='discounted factor')
    parser.add_argument('--tau', type = float, default = 1.0, help='temperature')
    parser.add_argument('--ep-len', type = int, default = 1000, help='horizon length')
    
    parser.add_argument('--POMDP', action='store_true', default=False, help='whether use partial observation')
    parser.add_argument('--obs-noise', type=float, default=0.1)

    parser.add_argument('--PO-type', type=str, default='noise', choices=['noise', 'mask'], help='how to create observation')
    parser.add_argument('--mask-index', type=int, nargs='+', default=[0])
    
    parser.add_argument('--log-dir', type = str, default = 'OnPolicy', help='directory for log')
    args = parser.parse_args()

    return args

def main(args):
    env_name = "CartPole"
    ep_len = args.ep_len
    seed = args.seed
    U.set_seed(seed)

    env = CartPoleEnv(max_ep_len=ep_len,
            seed=seed + 1000,
            partial_obs=args.POMDP, 
            partial_obs_type=args.PO_type,
            mask_index=args.mask_index,
            obs_noise=args.obs_noise)

    obs_dim = 4
    act_dim = 2

    sess = U.make_session()
    sess.__enter__()

    '''load evaluation policy'''
    q_net = Q_network(obs_dim, act_dim, seed=args.seed + 2000, default_tau=args.tau)
    U.initialize_all_vars()

    if args.PO_type == 'noise':
        model_dir = './CartPole_Model/PO_Model_Noise_0.1_Expert/Model'  
    elif args.PO_type == 'mask':
        model_dir = './CartPole_Model/Reward-2500/Model'  
    else:
        raise NotImplementedError

    q_net.load_model(model_dir)
    
    avg_ep_rews, ep_rews = U.eval_policy_cartpole(env, q_net, ep_num=100, gamma=args.gamma)

    print(avg_ep_rews, np.std(ep_rews) / len(ep_rews))

    log_name = 'log.pickle'

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    if args.PO_type == 'noise':
        logger = ope_log_class(path=os.path.join(args.log_dir, 'CartPole_ObsNoise{}/Tau{}'.format(args.obs_noise, args.tau)), name=log_name, tau=args.tau, env_name=env_name, value_true=np.mean(ep_rews))
    elif args.PO_type == 'mask':
        logger = ope_log_class(path=os.path.join(args.log_dir, 'CartPole_MaskIndex{}/Tau{}'.format(args.mask_index, args.tau)), name=log_name, tau=args.tau, env_name=env_name, value_true=np.mean(ep_rews))
    else:
        raise NotImplementedError

    print(logger)
    logger.dump()

if __name__ == '__main__':
    args = get_parser()
    main(args)
