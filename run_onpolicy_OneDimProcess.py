import numpy as np
import pickle
import os
import argparse
from Env.OneDimProcess import Process, Sigmoid_Policy


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', type = float, nargs='+', default = [1.0], help='w in sigmoid policy')
    parser.add_argument('-b', type = float, default = 1.0, help='b in sigmoid policy')
    
    parser.add_argument('--delta', type = float, default = 1.0, help='delta in dynamic')
    parser.add_argument('--inherent-noise', type = float, default = 0.5, help='inherent noise in transition')

    parser.add_argument('--obs-noise', type = float, nargs='+', default = [0.5], help='noise level for partial observation')

    parser.add_argument('--gamma', type = float, default = 0.95, help='discounted factor')
    parser.add_argument('--ep-len', type = int, default = 100, help='episode length')

    parser.add_argument('--log-dir', type = str, default = 'OnPolicy', help='directory for log')
    args = parser.parse_args()

    return args

def test(policy, delta, obs_noise, ep_len, inherent_noise, gamma):
    POMDP = obs_noise > 0.0
    env = Process(delta=delta, 
            obs_noise=obs_noise, 
            POMDP=POMDP, max_ep_len=ep_len, 
            inherent_noise=inherent_noise)

    init_obs = env.reset()
    obs = init_obs
    accum_return = 0.0
    factor = 1.0
    for i in range(ep_len):
        act = policy.sample_action(obs)
        next_obs, reward, done, info = env.step(act)

        accum_return += reward * factor
        factor *= gamma
        obs = next_obs

    return accum_return


if __name__ == '__main__':
    args = get_parser()

    gamma = args.gamma
    delta = args.delta
    ep_len = args.ep_len
    inherent_noise = args.inherent_noise
    b = args.b

    OnPolicy_Records = {}

    for obs_noise in args.obs_noise:
        OnPolicy_Records[obs_noise] = {}
        for w in args.w:
            policy = Sigmoid_Policy(w=w, b=b)

            POMDP_accum_return = []
            num_tests = 200
            for i in range(num_tests):
                POMDP_accum_return.append(
                    test(policy=policy, ep_len=ep_len, delta=delta, obs_noise=obs_noise, inherent_noise=inherent_noise, gamma=gamma)
                )

            mean, std_err = np.mean(POMDP_accum_return), np.std(POMDP_accum_return) / len(POMDP_accum_return)
            print('Evaluation Results of Sigmoid Policy with w={}, b={}, ObsNoise={}'.format(w, b, obs_noise))
            print('Mean: {}, Standard Error {}'.format(mean, std_err))

            OnPolicy_Records[obs_noise][w] = {'Mean': mean, 'Std_Err': std_err}

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    log_file_path = os.path.join(args.log_dir, 'OneDimProcess_OnPolicy_ep{}-delta{}-ObsNoise{}-InNoise{}_w{}_b{}'.format(args.ep_len, args.delta, args.obs_noise, args.inherent_noise, args.w, args.b))

    with open(log_file_path, 'wb') as f:
        pickle.dump(OnPolicy_Records, f)

    print(os.getcwd())