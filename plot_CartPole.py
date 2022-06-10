import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obs-noise', type = float, default = 0.1, help='size of obs noise')

    parser.add_argument('--sample-size', type=int, default=200000)

    parser.add_argument('--plot-bias', action='store_true', default=False, help='whether to plot the bias')

    parser.add_argument('--baseline-log-dir', type=str, default='./log', help='The default log path')
    parser.add_argument('--PO-log-dir', type=str, default='./log', help='The default log path')
    
    parser.add_argument('--target-tau', type=float, nargs='+', default=[1.0])

    parser.add_argument('--PO-type', type=str, default='noise', choices=['noise', 'mask'], help='how to create observation')
    parser.add_argument('--mask-index', type=int, nargs='+', default=[0])

    args = parser.parse_args()

    return args


def main():
    args = get_parser()
    
    obs_noise = args.obs_noise
    sample_size = args.sample_size

    # log_dirs = {
    #     'Baseline': './log/ObsNoise{}_Size{}/Est_DR_Baseline'.format(args.obs_noise, args.sample_size), 
    #     'PO_Est': './log/ObsNoise{}_Size{}/Est_DR'.format(args.obs_noise, args.sample_size)
    # }
    log_dirs = {
        'Baseline': args.baseline_log_dir,
        'PO_Est': args.PO_log_dir,
    }
    base_path = os.getcwd()

    AllMethods_AllEstResults = {}
    
    for method in ['Baseline', 'PO_Est']:
        os.chdir(base_path)
        ld = log_dirs[method]
        os.chdir(ld)
        AllEstResults = {}
        AllMethods_AllEstResults[method] = AllEstResults

        print(method, os.listdir(), os.getcwd())

        # under current directory, dir name is 'Dataset{}'.format(dataset_seed)
        for d in os.listdir():
            os.chdir(d)
            for tau in args.target_tau:
                # load true value
                if args.PO_type == 'noise':
                    env_name = 'CartPole_ObsNoise{}'.format(args.obs_noise)
                else:
                    env_name = 'CartPole_MaskIndex{}'.format(args.mask_index)
                with open(os.path.join(base_path, 'OnPolicy', env_name, 'Tau{}'.format(tau), 'log.pickle'), 'rb') as f:
                    data = pickle.load(f)
                    True_Value = data['True_Rew']

                if tau not in AllEstResults.keys():
                    AllEstResults[tau] = {}
                    AllEstResults[tau]['True_Value'] = True_Value
                    AllEstResults[tau]['(PO-)MQL'] = []
                    AllEstResults[tau]['(PO-)MWL'] = []
                    AllEstResults[tau]['(PO-)DR'] = []

                for file_name in os.listdir():
                    if not file_name.endswith('Tau{}.pickle'.format(tau)):
                        continue
                    with open(file_name, 'rb') as f:
                        data = pickle.load(f)

                        # load data from iter 9000 to 10000
                        start = 9000
                        end = 10000
                        PO_MWL_value, PO_MQL_value, PO_DR_value = [], [], []
                        for i in range(len(data['(PO-)MWL'])):
                            if data['(PO-)MWL'][i][0] > start and data['(PO-)MWL'][i][0] < end:
                                PO_MWL_value.append(data['(PO-)MWL'][i][-1])
                                PO_MQL_value.append(data['(PO-)MQL'][i][-1])
                                PO_DR_value.append(data['(PO-)DR'][i][-1])

                    AllEstResults[tau]['(PO-)MQL'].append(np.mean(PO_MQL_value))
                    AllEstResults[tau]['(PO-)MWL'].append(np.mean(PO_MWL_value))
                    AllEstResults[tau]['(PO-)DR'].append(np.mean(PO_DR_value))
            os.chdir('..')

        AllEstResults[tau]['(PO-)MQL'] = np.array(AllEstResults[tau]['(PO-)MQL'])
        AllEstResults[tau]['(PO-)MWL'] = np.array(AllEstResults[tau]['(PO-)MWL'])
        AllEstResults[tau]['(PO-)DR'] = np.array(AllEstResults[tau]['(PO-)DR'])

    print('\n\n\n')
    for k in AllMethods_AllEstResults.keys():
        for tau in args.target_tau:
            print(tau, AllMethods_AllEstResults[k][tau])
    print('\n\n\n')

    # use w as x_coord
    x_coord = list(args.target_tau)
    x_coord.sort()

    Baseline_Records = {}
    PO_Est_Records = {}

    for est in ['MQL', 'MWL', 'DR']:
        est_name = '(PO-)' + est

        # used for compute bias
        Baseline_Bias = {}
        PO_Bias = {}    
        Baseline_Bias_StdErr = {}
        PO_Bias_StdErr = {}    
        Baseline_Bias_ErrBar = {}
        PO_Bias_ErrBar = {}    

        # used for compute MSE
        Baseline_Mean = {}
        PO_Mean = {}
        Baseline_MSE = {}
        PO_MSE = {}
        Baseline_MSE_StdErr = {}
        PO_MSE_StdErr = {}
        Baseline_MSE_ErrBar = {}
        PO_MSE_ErrBar = {}
        
        # compute average bias and std err
        for tau in x_coord:
            True_Value = AllMethods_AllEstResults['Baseline'][tau]['True_Value']

            print(AllMethods_AllEstResults['Baseline'][tau])

            # compute MSE and error bar
            Baseline_Mean[tau] = np.mean(AllMethods_AllEstResults['Baseline'][tau][est_name])
            PO_Mean[tau] = np.mean(AllMethods_AllEstResults['PO_Est'][tau][est_name])

            Baseline_MSE[tau] = np.mean(np.square(AllMethods_AllEstResults['Baseline'][tau][est_name] / True_Value - 1.0))
            PO_MSE[tau] = np.mean(np.square(AllMethods_AllEstResults['PO_Est'][tau][est_name] / True_Value - 1.0))

            Baseline_MSE_StdErr[tau] = np.std(np.square(AllMethods_AllEstResults['Baseline'][tau][est_name] / True_Value - 1.0), ddof=1) / np.sqrt(len(AllMethods_AllEstResults['Baseline'][tau][est_name]))
            PO_MSE_StdErr[tau] = np.std(np.square(AllMethods_AllEstResults['PO_Est'][tau][est_name] / True_Value - 1.0), ddof=1) / np.sqrt(len(AllMethods_AllEstResults['PO_Est'][tau][est_name]))

            Baseline_MSE_ErrBar[tau] = [
                np.log(Baseline_MSE[tau] + 2 * Baseline_MSE_StdErr[tau]) - np.log(Baseline_MSE[tau]), 
                np.log(Baseline_MSE[tau]) - np.log(Baseline_MSE[tau] - 2 * Baseline_MSE_StdErr[tau]),
            ]
            PO_MSE_ErrBar[tau] = [
                np.log(PO_MSE[tau] + 2 * PO_MSE_StdErr[tau]) - np.log(PO_MSE[tau]), 
                np.log(PO_MSE[tau]) - np.log(PO_MSE[tau] - 2 * PO_MSE_StdErr[tau]),
            ]

            # compute the average bias
            Baseline_Bias[tau] = np.abs(np.mean(AllMethods_AllEstResults['Baseline'][tau][est_name] / True_Value - 1.0))
            PO_Bias[tau] = np.abs(np.mean(AllMethods_AllEstResults['PO_Est'][tau][est_name] / True_Value - 1.0))

            # compute standard error of the bias
            Baseline_Bias_StdErr[tau] = np.std(AllMethods_AllEstResults['Baseline'][tau][est_name] / True_Value - 1.0, ddof=1) / np.sqrt(len(AllMethods_AllEstResults['Baseline'][tau][est_name]))
            PO_Bias_StdErr[tau] = np.std(AllMethods_AllEstResults['PO_Est'][tau][est_name] / True_Value - 1.0, ddof=1) / np.sqrt(len(AllMethods_AllEstResults['PO_Est'][tau][est_name]))   

            # compute the error bar
            Baseline_Bias_ErrBar[tau] = [
                np.log(Baseline_Bias[tau] + 2 * Baseline_Bias_StdErr[tau]) - np.log(Baseline_Bias[tau]),
                np.log(Baseline_Bias[tau]) - np.log(Baseline_Bias[tau] - 2 * Baseline_Bias_StdErr[tau]),
            ]
            PO_Bias_ErrBar[tau] = [
                np.log(PO_Bias[tau] + 2 * PO_Bias_StdErr[tau]) - np.log(PO_Bias[tau]),
                np.log(PO_Bias[tau]) - np.log(PO_Bias[tau] - 2 * PO_Bias_StdErr[tau]),
            ]  

        Baseline_Records[est] = {
            'MSE': Baseline_MSE,
            'MSE_ErrBar': Baseline_MSE_ErrBar,
            'Bias': Baseline_Bias,
            'Bias_ErrBar': Baseline_Bias_ErrBar,
        }
        PO_Est_Records[est] = {
            'MSE': PO_MSE,
            'MSE_ErrBar': PO_MSE_ErrBar,
            'Bias': PO_Bias,
            'Bias_ErrBar': PO_Bias_ErrBar,
        }

    fontsize = 20
    linewidth = 3.0
    capsize = 10.0
    plt.figure(figsize=(7, 6), tight_layout=True)

    print(Baseline_Records)

    if args.plot_bias:
        for estimator, marker, color in zip(['MQL', 'MWL', 'DR'], ['o', 's', 'D'], ['r', 'b', 'g']):
            Baseline_log_Bias = np.log([Baseline_Records[estimator]['Bias'][tau] for tau in x_coord])
            PO_log_Bias = np.log([PO_Est_Records[estimator]['Bias'][tau] for tau in x_coord])

            Baseline_ErrBar = np.array([Baseline_Records[estimator]['Bias_ErrBar'][tau] for tau in x_coord]).transpose()
            PO_ErrBar = np.array([PO_Est_Records[estimator]['Bias_ErrBar'][tau] for tau in x_coord]).transpose()

            plt.errorbar(range(len(x_coord)), Baseline_log_Bias, yerr=Baseline_ErrBar, color=color, label=estimator, marker=marker, linestyle='--', capsize=capsize, linewidth=linewidth, markersize=10)
            plt.errorbar(range(len(x_coord)), PO_log_Bias, yerr=PO_ErrBar, color=color, label='(PO-)' + estimator, marker=marker, linestyle='-', capsize=capsize, linewidth=linewidth, markersize=10)
            plt.ylabel('Log of Bias (relative)', fontsize=fontsize)
    else:
        for estimator, marker, color in zip(['MQL', 'MWL', 'DR'], ['o', 's', 'D'], ['r', 'b', 'g']):
            Baseline_log_MSE = np.log([Baseline_Records[estimator]['MSE'][tau] for tau in x_coord])
            PO_log_MSE = np.log([PO_Est_Records[estimator]['MSE'][tau] for tau in x_coord])

            Baseline_ErrBar = np.array([Baseline_Records[estimator]['MSE_ErrBar'][tau] for tau in x_coord]).transpose()
            PO_ErrBar = np.array([PO_Est_Records[estimator]['MSE_ErrBar'][tau] for tau in x_coord]).transpose()

            plt.errorbar(range(len(x_coord)), Baseline_log_MSE, yerr=Baseline_ErrBar, color=color, label=estimator, marker=marker, linestyle='--', capsize=capsize, linewidth=linewidth, markersize=10)
            plt.errorbar(range(len(x_coord)), PO_log_MSE, yerr=PO_ErrBar, color=color, label='(PO-)' + estimator, marker=marker, linestyle='-', capsize=capsize, linewidth=linewidth, markersize=10)
            plt.ylabel('Log MSE (relative)', fontsize=fontsize)

    plt.legend(fontsize=fontsize, loc='best')
    plt.title(r'CartPole with Obs Noise $\sim \mathcal{N}(0,0.1)$', fontsize=fontsize+3)
    plt.xticks(range(len(args.target_tau)), args.target_tau, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel(r'Choice of $\tau_O$', fontsize=fontsize)
    
    os.chdir(base_path)
    
    if not os.path.exists('ExpFigures'):
        os.makedirs('ExpFigures')

    suffix = 'Bias' if args.plot_bias else 'MSE'
    plt.savefig('ExpFigures/CartPole_N{}_{}.png'.format(args.obs_noise, suffix))
    plt.show()


if __name__ == '__main__':
    main()