# Introduction
Code for Paper 


> A Minimax Learning Approach to Off-Policy Evaluation in Confounded Partially Observable Markov Decision Processes.
>
> Chengchun Shi\*, Masatoshi Uehara\*, Jiawei Huang, Nan Jiang. ICML 2022. (*: equal contribution).
>
> [https://arxiv.org/abs/2111.06784](https://arxiv.org/abs/2111.06784).

We consider off-policy evaluation (OPE) in Partially Observable Markov Decision Processes (POMDPs),
where the evaluation policy depends only on observable variables and the behavior policy depends on unobservable
latent variables. Existing works either assume no unmeasured confounders, or focus on settings where both the
observation and the state spaces are tabular. In this work, we first propose novel identification methods for OPE
in POMDPs with latent confounders, by introducing bridge functions that link the target policy’s value and the
observed data distribution. We next propose minimax estimation methods for learning these bridge functions, and
construct three estimators based on these estimated bridge functions, corresponding to a value function-based
estimator, a marginalized importance sampling estimator, and a doubly-robust estimator. Our proposal permits
general function approximation and is thus applicable to settings with continuous or large observation/state spaces.
The nonasymptotic and asymptotic properties of the proposed estimators are investigated in detail.

If you find it helpful, please cite as follow:
```
@misc{shi2021minimax,
    title={A Minimax Learning Approach to Off-Policy Evaluation in Confounded Partially Observable Markov Decision Processes},
    author={Chengchun Shi and Masatoshi Uehara and Jiawei Huang and Nan Jiang},
    year={2021},
    eprint={2111.06784},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```




# Instructions for running experiments
## Environments

```
python 3.6.13
tensorflow==1.15.0
numpy==1.19.5
gym==0.21.0
sklearn==0.24.2
```

## Illustration of the Bias of Baseline Methods
```python
# case 1
python binary_example.py --behavior-eps 0.25
# case 2
python binary_example.py --behavior-eps 0.5
# case 3
python binary_example.py --behavior-eps 0.75
```
One can observe from the running output that, except for case 2 (no confounding issue), in case 1 & 3, the baseline least-square estimator will produce biased results, while our methods can produce unbiased results for all the cases.

## Experiments on Toy Example
### Varying Dynamic Noise Level 

```python
# Generate the dataset by behavior policy
python GeneData_OneDimProcess.py -w -1.0 -b 1.0 --obs-noise 0.5 --dataset-seed 0 100 200 300 400 500 600 700 800 900
python GeneData_OneDimProcess.py -w -1.0 -b 1.0 --obs-noise 1.0 --dataset-seed 0 100 200 300 400 500 600 700 800 900
python GeneData_OneDimProcess.py -w -1.0 -b 1.0 --obs-noise 1.5 --dataset-seed 0 100 200 300 400 500 600 700 800 900

# Compute true value by on-policy samples
python run_onpolicy_OneDimProcess.py --obs-noise 0.5 1.0 1.5 -w -3.0 -2.0 1.0 2.0

# Run experiments with generated dataset
python run_solve_linear_toy.py -w -3.0 -2.0 1.0 2.0 --feature-dim 100 --gamma 0.95 --obs-noise 0.5 --seed 1000 2000 3000 4000 5000 --dataset-seed 0 100 200 300 400 500 600 700 800 900
python run_solve_linear_toy.py -w -3.0 -2.0 1.0 2.0 --feature-dim 100 --gamma 0.95 --obs-noise 1.0 --seed 1000 2000 3000 4000 5000 --dataset-seed 0 100 200 300 400 500 600 700 800 900
python run_solve_linear_toy.py -w -3.0 -2.0 1.0 2.0 --feature-dim 100 --gamma 0.95 --obs-noise 1.5 --seed 1000 2000 3000 4000 5000 --dataset-seed 0 100 200 300 400 500 600 700 800 900

# Plot results
## Plot MSE
python plot_toy.py --target-w -3.0 -2.0 1.0 2.0 --sample-size 200000 --obs-noise 1.0
## Plot bias
python plot_toy.py --target-w -3.0 -2.0 1.0 2.0 --sample-size 200000 --obs-noise 1.0 --plot-bias
```

### Varying Sample Size
```python
# Generate the dataset by behavior policy
python GeneData_OneDimProcess.py -w -1.0 -b 1.0 --obs-noise 1.0 --dataset-seed 0 100 200 300 400 500 600 700 800 900 --sample-size 50000
python GeneData_OneDimProcess.py -w -1.0 -b 1.0 --obs-noise 1.0 --dataset-seed 0 100 200 300 400 500 600 700 800 900 --sample-size 100000
python GeneData_OneDimProcess.py -w -1.0 -b 1.0 --obs-noise 1.0 --dataset-seed 0 100 200 300 400 500 600 700 800 900 --sample-size 200000

# Compute true value by on-policy samples
python run_onpolicy_OneDimProcess.py --obs-noise 0.5 1.0 1.5 -w -3.0 -2.0 1.0 2.0

# Run experiments with generated dataset
python run_solve_linear_toy.py -w -3.0 -2.0 1.0 2.0 --feature-dim 100 --gamma 0.95 --obs-noise 1.0 --seed 1000 2000 3000 4000 5000 --dataset-seed 0 100 200 300 400 500 600 700 800 900 --sample-size 50000
python run_solve_linear_toy.py -w -3.0 -2.0 1.0 2.0 --feature-dim 100 --gamma 0.95 --obs-noise 1.0 --seed 1000 2000 3000 4000 5000 --dataset-seed 0 100 200 300 400 500 600 700 800 900 --sample-size 100000
python run_solve_linear_toy.py -w -3.0 -2.0 1.0 2.0 --feature-dim 100 --gamma 0.95 --obs-noise 1.0 --seed 1000 2000 3000 4000 5000 --dataset-seed 0 100 200 300 400 500 600 700 800 900 --sample-size 200000

# Plot results
## Plot MSE
python plot_toy_vary_sample_size.py --target-w -3.0 -2.0 1.0 2.0 --sample-size 50000 100000 200000 --obs-noise 1.0
## Plot bias
python plot_toy_vary_sample_size.py --target-w -3.0 -2.0 1.0 2.0 --sample-size 50000 100000 200000 --obs-noise 1.0 --plot-bias
```


## Experiments on CartPole
The partial observation can be created by adding noise on or mask out some dimension of the orginal full observation. Our code support both implementations and the only difference in running code is specify `--PO-noise noise` for the former and `--PO-noise mask` for the latter.
### Prepare Behavior & Target Policy for CartPole (Optional)

We provide the well-trained policy (TF Model) in the `./CartPole_Model` directory. If you want to generate your own, you can follow this instructions:
```python
# Step 1: We build our policy generating code based on OpenAI baselines. So first download baselines
git clone https://github.com/openai/baselines.git
git checkout 1f3c3e3

# Step 2: Install the baselines follow its official instructions

# Step 3: Copy updated files to baselines' directory
cd ./baselines/baselines/deepq
rsync -a [path of this code]/GenerateModel/* ./
rsync -a [path of this code]/Env/* ./

# Step 4: Run DQN to generate policy. You may set your own training hyper-parameters
python train_Cartpole.py --PO-type noise #(mask)

# Step 5: Move new model to replace the old one
mv ./CartPole-v0 [path of this code]/CartPole_Model
```

### Running Experiments

```python
# Generate the dataset by behavior policy
python GeneData_CartPole.py --behavior-tau 5.0 --POMDP --obs-noise 0.1 --dataset-seed 0 100 200 300 400 500 600 700 800 900  --PO-type noise #(mask)

# Compute true value by on-policy samples
python run_onpolicy_CartPole.py --gamma 0.95 --tau 3.0 --POMDP --obs-noise 0.1 --PO-type noise #(mask)
python run_onpolicy_CartPole.py --gamma 0.95 --tau 4.0 --POMDP --obs-noise 0.1 --PO-type noise #(mask)
python run_onpolicy_CartPole.py --gamma 0.95 --tau 5.0 --POMDP --obs-noise 0.1 --PO-type noise #(mask)
python run_onpolicy_CartPole.py --gamma 0.95 --tau 6.0 --POMDP --obs-noise 0.1 --PO-type noise #(mask)

# Evaluate PO-MWL, PO-MQL and PO-DR (Batch Run)
python run_solve_dr_RKHS.py --POMDP --target-tau 3.0 4.0 5.0 6.0 --obs-noise 0.1 --gamma 0.95 --norm std_norm --seed 100 --dataset-seed 0 100 200 300 400 500 600 700 800 900 --iter 10000 --kernel-bw-tau 0.5 --kernel-bv-tau 0.2 --PO-type noise #(mask)

# Evaluate MWL, MQL, DR (Batch Run)
python run_solve_dr_RKHS.py --POMDP --target-tau 3.0 4.0 5.0 6.0 --obs-noise 0.1 --gamma 0.95 --norm std_norm --seed 100 --dataset-seed 0 100 200 300 400 500 600 700 800 900 --iter 10000 --kernel-bw-tau 0.5 --kernel-bv-tau 0.2 --baseline --PO-type noise #(mask)

# Plot results
You need to specify the log directories for experiments of baseline methods and our PO methods after `--baseline-log-dir` and `--PO-log-dir`, respectively.
## Plot MSE
python plot_CartPole.py --target-tau 3.0 4.0 5.0 6.0 --baseline-log-dir ... --PO-log-dir ...
## Plot Bias
python plot_CartPole.py --target-tau 3.0 4.0 5.0 6.0 --plot-bias --baseline-log-dir ... --PO-log-dir ...
```
