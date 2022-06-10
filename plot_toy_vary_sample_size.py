import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import argparse

'''
python .\plot_toy.py --noise-level 1.0 --feature-dim 25 --same-seed
'''

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obs-noise', type = float, default = 1.0, help='size of obs noise')
    parser.add_argument('--target-w', type = float, nargs='+', default = [1.0], help='choice of w of target policy')
    parser.add_argument('--target-b', type = float, default = 1.0, help='choice of b of target policy')

    parser.add_argument('--feature-dim', type=int, default=100)
    parser.add_argument('--inherent-noise', type = float, default = 0.5, help='inherent noise level')

    parser.add_argument('--alpha', type = float, default = 5.0, help='choice RBF kernel sampler')

    parser.add_argument('--plot-bias', action='store_true', default=False, help='whether to plot the bias')

    parser.add_argument('--true-value-log-path', type=str, default=None, help='The path to the log of true value')
    
    parser.add_argument('--sample-size', nargs='+', type=int, default=[200000])

    args = parser.parse_args()

    return args


def main():
    args = get_parser()
    
    color_list = ['r', 'b', 'g', 'purple', 'orange']

    obs_noise = args.obs_noise

    sample_size = args.sample_size

    if args.true_value_log_path is None:
        true_value_log_path = './OnPolicy/OneDimProcess_OnPolicy_ep100-delta1.0-ObsNoise[0.5, 1.0, 1.5]-InNoise0.5_w[-3.0, -2.0, 1.0, 2.0]_b1.0'

    with open(true_value_log_path, 'rb') as f:
        True_Value_Table = pickle.load(f)

    # we clip those abnormal values
    upper_clip = 50.0
    lower_clip = -0.

    All_SS_EstResults = {}

    # use w as x_coord
    x_coord = args.sample_size
    x_coord.sort()

    for ss in args.sample_size:
        AllEstResults = {}
        All_SS_EstResults[ss] = {}
        All_SS_EstResults[ss]['AllEstResults'] = AllEstResults
        
        log_path = './log/Toy_Linear_Est_DR/FDim{}_ObsNoise{}_InNoise{}_Alpha{}_S{}.pickle'.format(args.feature_dim, obs_noise, args.inherent_noise, args.alpha, ss)

        with open(log_path, 'rb') as f:
            log_data = pickle.load(f)
            
        for dataset_seed in log_data.keys():
            for w in args.target_w:
                assert True_Value_Table[args.obs_noise][w] is not None, 'True Value hasn\'t been computed for w={}'.format(w)
                True_Value = True_Value_Table[args.obs_noise][w]['Mean']
                if w not in AllEstResults.keys():
                    AllEstResults[w] = {
                        'Baseline': [],
                        'PO_Est': [],
                        'True_Value': True_Value,
                    }
                    print('Append ', w, ' into keys ')

                PO_with_diff_RBF_seed = []
                Baseline_with_diff_RBF_seed = []
                for seed in log_data[dataset_seed][w].keys():
                    if log_data[dataset_seed][w][seed]['PO_estimator'] >= lower_clip and log_data[dataset_seed][w][seed]['PO_estimator'] <= upper_clip:
                        PO_with_diff_RBF_seed.append(log_data[dataset_seed][w][seed]['PO_estimator'])
                    else:
                        pass
                        
                    if log_data[dataset_seed][w][seed]['Baseline_estimator'] >= lower_clip and log_data[dataset_seed][w][seed]['Baseline_estimator'] <= upper_clip:
                        Baseline_with_diff_RBF_seed.append(log_data[dataset_seed][w][seed]['Baseline_estimator'])
                    else:
                        pass

                PO_for_this_dataset = np.mean(PO_with_diff_RBF_seed)
                Baseline_for_this_dataset = np.mean(Baseline_with_diff_RBF_seed)

                AllEstResults[w]['PO_Est'].append(PO_for_this_dataset)
                AllEstResults[w]['Baseline'].append(Baseline_for_this_dataset)

        for w in args.target_w:
            AllEstResults[w]['PO_Est'] = np.array(AllEstResults[w]['PO_Est'])
            AllEstResults[w]['Baseline'] = np.array(AllEstResults[w]['Baseline'])

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
        
        All_SS_EstResults[ss]['Baseline_Bias'] = Baseline_Bias
        All_SS_EstResults[ss]['PO_Bias'] = PO_Bias
        All_SS_EstResults[ss]['Baseline_Bias_ErrBar'] = Baseline_Bias_ErrBar
        All_SS_EstResults[ss]['PO_Bias_ErrBar'] = PO_Bias_ErrBar

        All_SS_EstResults[ss]['Baseline_MSE'] = Baseline_MSE
        All_SS_EstResults[ss]['PO_MSE'] = PO_MSE
        All_SS_EstResults[ss]['Baseline_MSE_ErrBar'] = Baseline_MSE_ErrBar
        All_SS_EstResults[ss]['PO_MSE_ErrBar'] = PO_MSE_ErrBar

        True_Value_list = []
        # compute average bias and std err
        for w in args.target_w:
            True_Value = AllEstResults[w]['True_Value']
            True_Value_list.append(True_Value)

            # compute MSE and error bar
            Baseline_Mean[w] = np.mean(AllEstResults[w]['Baseline'])
            PO_Mean[w] = np.mean(AllEstResults[w]['PO_Est'])

            Baseline_MSE[w] = np.mean(np.square(AllEstResults[w]['Baseline'] / True_Value - 1.0))
            PO_MSE[w] = np.mean(np.square(AllEstResults[w]['PO_Est'] / True_Value - 1.0))

            Baseline_MSE_StdErr[w] = np.std(np.square(AllEstResults[w]['Baseline'] / True_Value - 1.0), ddof=1) / np.sqrt(len(AllEstResults[w]['Baseline']))
            PO_MSE_StdErr[w] = np.std(np.square(AllEstResults[w]['PO_Est'] / True_Value - 1.0), ddof=1) / np.sqrt(len(AllEstResults[w]['PO_Est']))

            Baseline_MSE_ErrBar[w] = [
                np.log(Baseline_MSE[w] + 2 * Baseline_MSE_StdErr[w]) - np.log(Baseline_MSE[w]), 
                np.log(Baseline_MSE[w]) - np.log(Baseline_MSE[w] - 2 * Baseline_MSE_StdErr[w]),
            ]
            PO_MSE_ErrBar[w] = [
                np.log(PO_MSE[w] + 2 * PO_MSE_StdErr[w]) - np.log(PO_MSE[w]), 
                np.log(PO_MSE[w]) - np.log(PO_MSE[w] - 2 * PO_MSE_StdErr[w]),
            ]

            # compute the average bias
            Baseline_Bias[w] = np.mean(AllEstResults[w]['Baseline'] / True_Value - 1.0)
            PO_Bias[w] = np.mean(AllEstResults[w]['PO_Est'] / True_Value - 1.0)

            # compute standard error of the bias
            Baseline_Bias_StdErr[w] = np.std(AllEstResults[w]['Baseline'] / True_Value - 1.0, ddof=1) / np.sqrt(len(AllEstResults[w]['Baseline']))
            PO_Bias_StdErr[w] = np.std(AllEstResults[w]['PO_Est'] / True_Value - 1.0, ddof=1) / np.sqrt(len(AllEstResults[w]['PO_Est']))   

            # compute the error bar
            Baseline_Bias_ErrBar[w] = [
                np.log(Baseline_Bias[w] + 2 * Baseline_Bias_StdErr[w]) - np.log(Baseline_Bias[w]),
                np.log(Baseline_Bias[w]) - np.log(Baseline_Bias[w] - 2 * Baseline_Bias_StdErr[w]),
            ]
            PO_Bias_ErrBar[w] = [
                np.log(PO_Bias[w] + 2 * PO_Bias_StdErr[w]) - np.log(PO_Bias[w]),
                np.log(PO_Bias[w]) - np.log(PO_Bias[w] - 2 * PO_Bias_StdErr[w]),
            ]  

    fontsize = 20
    linewidth = 3.0
    capsize = 10.0
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(15, 6))

    for w in args.target_w:
        color = color_list.pop(0)
        if args.plot_bias:
            Baseline_log_Bias = np.log([All_SS_EstResults[ss]['Baseline_Bias'][w] for ss in args.sample_size])
            PO_log_Bias = np.log([All_SS_EstResults[ss]['PO_Bias'][w] for ss in args.sample_size])

            print(Baseline_log_Bias)
            print(PO_log_Bias)

            Baseline_ErrBar = np.array([All_SS_EstResults[ss]['Baseline_Bias_ErrBar'][w] for ss in args.sample_size]).transpose()
            PO_ErrBar = np.array([All_SS_EstResults[ss]['PO_Bias_ErrBar'][w] for ss in args.sample_size]).transpose()

            axs[0].errorbar(range(len(x_coord)), Baseline_log_Bias, yerr=Baseline_ErrBar, color=color, label='Baseline w={}'.format(w), marker='o', capsize=capsize, linewidth=linewidth, linestyle='--', markersize=10)
            axs[1].errorbar(range(len(x_coord)), PO_log_Bias, yerr=PO_ErrBar, color=color, linestyle='-', label='Ours w={}'.format(w), marker='s', capsize=capsize, linewidth=linewidth, markersize=10)
        else:
            Baseline_log_MSE = np.log([All_SS_EstResults[ss]['Baseline_MSE'][w] for ss in args.sample_size])
            PO_log_MSE = np.log([All_SS_EstResults[ss]['PO_MSE'][w] for ss in args.sample_size])

            print(Baseline_log_MSE)
            print(PO_log_MSE)
            
            Baseline_ErrBar = np.array([All_SS_EstResults[ss]['Baseline_MSE_ErrBar'][w] for ss in args.sample_size]).transpose()
            PO_ErrBar = np.array([All_SS_EstResults[ss]['PO_MSE_ErrBar'][w] for ss in args.sample_size]).transpose()

            axs[0].errorbar(range(len(x_coord)), Baseline_log_MSE, yerr=Baseline_ErrBar, label='Baseline w={}'.format(w), marker='o', capsize=capsize, linewidth=linewidth, markersize=10, linestyle='--', color=color)
            axs[1].errorbar(range(len(x_coord)), PO_log_MSE, yerr=PO_ErrBar, label='Ours w={}'.format(w), marker='s', capsize=capsize, linewidth=linewidth, markersize=10, linestyle='-', color=color)

    title = 'One-Dim Process (Varying Sample Size)'.format(args.obs_noise, args.feature_dim)

    plt.suptitle(title, fontsize=fontsize + 1)
    handles, labels = [(a + b) for a, b in zip(axs[0].get_legend_handles_labels(), axs[1].get_legend_handles_labels())]
    fig.legend(handles, labels, bbox_to_anchor=(1.0,0.7), fontsize=fontsize)
    for i in range(len(axs)):
        ax = axs[i]
        xticks = ['5e4', '1e5', '2e5']
        ax.set_xticks(range(len(x_coord)))
        ax.set_xticklabels(xticks, fontsize=fontsize)
        y_ticks = ax.get_yticks()
        ax.set_yticklabels(y_ticks, fontsize=fontsize)
        ax.set_xlabel('Sample Size', fontsize=fontsize)
        
        if i == 0:
            if args.plot_bias:
                ax.set_ylabel('Log of Bias (relative)', fontsize=fontsize)
            else:
                ax.set_ylabel('Log MSE (relative)', fontsize=fontsize)

    if not os.path.exists('ExpFigures'):
        os.makedirs('ExpFigures')

    suffix = 'Bias' if args.plot_bias else 'MSE'
    plt.savefig('ExpFigures/Toy_VarySS_{}.png'.format('Bias' if args.plot_bias else 'MSE'))
    plt.show()


if __name__ == '__main__':
    main()