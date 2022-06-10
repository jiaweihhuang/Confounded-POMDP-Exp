import numpy as np
from sklearn.kernel_approximation import RBFSampler
import utils as U

'''
dataset:
'''
class Feature_Dataset():
    def __init__(self, dataset, 
            num_acts, feature_dim, 
            alpha, feature_type='rbf',
            seed=0):
        self.dataset = dataset
        self.num_acts = num_acts
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.feature_type = feature_type

        self.feature = {
            'last_obs_acts': None,
            'obs_acts': None,
            'next_obs_acts': [None] * self.num_acts,
            'init_obs_acts': [None] * self.num_acts,
        }

        # compute feature of last_obs_acts, feature.shape = [None, feature_dim]
        assert self.feature_dim % num_acts == 0, 'self.feature dim % num_acts != 0'
        n_components = self.feature_dim // num_acts

        # compute feature of obs_acts
        if self.feature_type == 'rbf':
            feature_bv = RBFSampler(gamma=self.alpha, random_state=seed + 10, n_components=n_components)
        else:
            feature_bv = None
        all_obs_list, all_acts_list = [], []

        all_obs_list.append(dataset['obs'])
        all_acts_list.append(dataset['acts'])
        for act in range(self.num_acts):
            all_obs_list.append(dataset['next_obs'])
            all_acts_list.append(act * np.ones(shape=[dataset['next_obs'].shape[0], 1]))
        for act in range(self.num_acts):
            all_obs_list.append(dataset['init_obs'])
            all_acts_list.append(act * np.ones(shape=[dataset['init_obs'].shape[0], 1]))

        all_obs_list.append(dataset['last_obs'])
        all_acts_list.append(dataset['acts'])

        all_obs = np.concatenate(all_obs_list, axis=0)
        all_acts = np.concatenate(all_acts_list, axis=0)
        all_features = self.create_features(all_obs, all_acts, feature_bv)

        print(all_features.shape)
        assert all_features.shape[1] == self.feature_dim, '{} != {}'.format(all_features.shape[1], self.feature_dim)

        data_size = self.dataset['obs'].shape[0]
        init_data_size = self.dataset['init_obs'].shape[0]
        self.feature['obs_acts'] = all_features[:data_size, :]
        start = data_size
        for act in range(self.num_acts):
            self.feature['next_obs_acts'][act] = all_features[start:start+data_size, :]
            start = start + data_size
        for act in range(self.num_acts):
            self.feature['init_obs_acts'][act] = all_features[start:start+init_data_size, :]
            start = start + init_data_size

        self.feature['last_obs_acts'] = all_features[start:start+data_size:, :]
        start += data_size

        assert start == all_features.shape[0], '{} != {}'.format(start, all_features.shape[0])

    def create_features(self, obs, act, rbf_feature):
        if self.feature_type == 'rbf':
            obs_feature = rbf_feature.fit_transform(obs)
        elif self.feature_type == 'one-hot':
            print(obs.astype(np.int32).shape)
            print(np.max(obs))
            print(np.max(obs.astype(np.int32)))
            obs_feature = np.eye(self.feature_dim // 2)[np.squeeze(obs).astype(np.int32)]
        elif self.feature_type == 'orthogonal':
            theta = np.random.rand * np.pi * 2
            orth_matrix = np.array([
                [np.sin(theta), np.cos(theta)], [np.cos(theta), -np.sin(theta)]
            ])
            obs_feature = orth_matrix[np.squeeze(obs).astype(np.int32)]
        else:
            raise NotImplementedError

        obs_act_feature = np.concatenate([obs_feature * (1.0 - act), obs_feature * act], axis=1)
        return obs_act_feature


class Linear_Class():
    def __init__(self, 
            obs_dim, 
            obs_ft_dim,
            num_acts,
            gamma,
            dataset,
            feature_dataset,
            ) -> None:

        self.gamma = gamma
        self.obs_dim = obs_dim
        self.obs_ft_dim = obs_ft_dim
        self.num_acts = num_acts

        self.dataset = dataset
        self.feature = feature_dataset.feature

        self.repara = True

        # the clipping threshold when computing Pseudo-inverse
        self.clip_threshold = 1e-6

    def learn(self):
        baseline_estimator = self.learn_baseline()
        estimator = self.learn_PO_Estimator()
        return estimator, baseline_estimator

    def learn_baseline(self):
        data_size = self.feature['obs_acts'].shape[0]

        # compute Pseudo-inverse
        matrix = np.matmul(self.feature['obs_acts'].transpose(), self.feature['obs_acts']) / data_size
        print(self.dataset['next_acts_probs'][0].shape)
        for act in range(self.num_acts):
            matrix -= self.gamma * np.matmul(self.feature['obs_acts'].transpose(), self.feature['next_obs_acts'][act] * self.dataset['next_acts_probs'][act]) / data_size
        self.pinv_matrix = np.linalg.pinv(matrix, rcond=self.clip_threshold)

        # compute init feature
        init_features = 0.0
        for act in range(self.num_acts):
            init_features += self.feature['init_obs_acts'][act] * self.dataset['init_acts_probs'][act]

        self.avg_init_feature = np.mean(init_features, axis=0, keepdims=True)

        # compute reward feature
        self.reward_feature = np.mean(self.dataset['rews'] * self.feature['obs_acts'], axis=0, keepdims=True)

        # compute final estimator
        self.baseline_estimator = np.matmul(np.matmul(self.avg_init_feature, self.pinv_matrix), self.reward_feature.transpose())

        return self.baseline_estimator


    def learn_PO_Estimator(self):
        data_size = self.feature['obs_acts'].shape[0]
        
        # compute Pseudo-inverse
        # recall that we reparameterize the bv with bv' * \pi^e
        acts_prob_Dot_feature = self.dataset['acts_probs'] * self.feature['last_obs_acts']
        matrix = np.matmul(acts_prob_Dot_feature.transpose(), self.feature['obs_acts']) / data_size

        for act in range(self.num_acts):
            matrix -= self.gamma * np.matmul(acts_prob_Dot_feature.transpose(), self.feature['next_obs_acts'][act] * self.dataset['next_acts_probs'][act]) / data_size

        self.pinv_matrix = np.linalg.pinv(matrix, rcond=self.clip_threshold)

        # compute init feature
        init_features = 0.0
        for act in range(self.num_acts):
            init_features += self.feature['init_obs_acts'][act] * self.dataset['init_acts_probs'][act]

        self.avg_init_feature = np.mean(init_features, axis=0, keepdims=True)

        # compute reward feature
        self.reward_feature = np.mean(self.dataset['rews'] * self.dataset['acts_probs'] * self.feature['last_obs_acts'], axis=0, keepdims=True)

        # compute final estimator
        self.estimator = np.matmul(np.matmul(self.avg_init_feature, self.pinv_matrix), self.reward_feature.transpose())

        return self.estimator