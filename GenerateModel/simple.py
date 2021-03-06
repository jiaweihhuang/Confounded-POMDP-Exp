import numpy as np
import os
import dill
import tempfile
import tensorflow as tf
import zipfile
import time

import baselines.common.tf_util as U

from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class ActWrapper(object):
    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params

    @staticmethod
    def load(path, num_cpu=16):
        with open(path, "rb") as f:
            model_data, act_params = dill.load(f)
        act = deepq.build_act(**act_params)
        sess = U.make_session(num_cpu=num_cpu)
        sess.__enter__()
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)

            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            U.load_state(os.path.join(td, "model"))

        return ActWrapper(act, act_params)

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def save(self, path):
        print(path)
        """Save model to a pickle located at `path`"""
        with tempfile.TemporaryDirectory() as td:
            U.save_state(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()
        with open(path, "wb") as f:
            dill.dump((model_data, self._act_params), f)


def load(path, num_cpu=16):
    """Load act function that was returned by learn function.

    Parameters
    ----------
    path: str
        path to the act function pickle
    num_cpu: int
        number of cpus to use for executing the policy

    Returns
    -------
    act: ActWrapper
        function that takes a batch of observations
        and returns actions.
    """
    return ActWrapper.load(path, num_cpu=num_cpu)


def learn(env,
          env_name,
          q_func,
          lr=2e-4,
          max_timesteps=10000000,
          buffer_size=50000,
          exploration_fraction=0.01,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          print_freq=1,
          checkpoint_freq=100,
          learning_starts=1000,
          gamma=1.0,
          target_network_update_freq=500,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          num_cpu=16,
          callback=None,
          eval=False):
    """Train a deepq model.

    Parameters
    -------
    env : gym.Env
        environment to train on
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    lr: float
        learning rate for adam optimizer
    max_timesteps: int
        number of env steps to optimizer for
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
        set to None to disable printing
    batch_size: int
        size of a batched sampled from replay buffer for training
    print_freq: int
        how often to print out training progress
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    prioritized_replay: True
        if True prioritized replay buffer will be used.
    prioritized_replay_alpha: float
        alpha parameter for prioritized replay buffer
    prioritized_replay_beta0: float
        initial value of beta for prioritized replay buffer
    prioritized_replay_beta_iters: int
        number of iterations over which beta will be annealed from initial value
        to 1.0. If set to None equals to max_timesteps.
    prioritized_replay_eps: float
        epsilon to add to the TD errors when updating priorities.
    num_cpu: int
        number of cpus to use for training
    callback: (locals, globals) -> None
        function called at every steps with state of the algorithm.
        If callback returns true training stops.

    Returns
    -------
    act: ActWrapper
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """
    # Create all the functions necessary to train the model

    sess = U.make_session(num_cpu=num_cpu)
    sess.__enter__()

    def make_obs_ph(name):
        return U.BatchInput(env.observation_space.shape, name=name)

    if env_name == "Pendulum-v0":
        act_dim = 7
        act_choice = [-1.5, -0.7, -0.3, 0, 0.3, 0.7, 1.5]
    else:
        act_dim = env.action_space.n
        act_choice = [i for i in range(act_dim)]

    act, train, update_target, debug = deepq.build_train(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=act_dim,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10,
    )
    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func,
        'num_actions': act_dim,
    }
    # Create the replay buffer
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = max_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None
    # Create the schedule for exploration starting from 1.
    if eval:
        learning_starts = max_timesteps
        exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                    initial_p=exploration_final_eps,
                                    final_p=exploration_final_eps)
    else:
        exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                    initial_p=1.0,
                                    final_p=exploration_final_eps)

    # Initialize the parameters and copy them to the target network.
    U.initialize()
    update_target()

    errors = [0.0 for i in range(10)]
    episode_rewards = [0.0]
    ep_len = [0]
    saved_mean_reward = None
    obs = env.reset()
    with tempfile.TemporaryDirectory() as td:
        model_saved = False
        # model_file = os.path.join(td, "model")
        
        model_file = os.path.join(env_name, "Model")
        if eval:
            print('load from ', model_file)
            U.load_state(model_file)

        for t in range(max_timesteps):
            
            # if callback is not None:
            #     if callback(locals(), globals()):
            #         break
            # Take action and update exploration to the newest value
            action_index = act(np.array(obs)[None], update_eps=exploration.value(t))[0]
            action = act_choice[action_index]
            # print(action)
            new_obs, rew, done, _ = env.step([action])
            # Store transition in the replay buffer.
            replay_buffer.add(obs, action_index, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            ep_len[-1] += 1
            if done:
                obs = env.reset()
                episode_rewards.append(0)
                ep_len.append(0)

            if t > learning_starts and t % train_freq == 0:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if prioritized_replay:
                    experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                    weights, batch_idxes = np.ones_like(rewards), None
                errors = train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                if prioritized_replay:
                    new_priorities = np.abs(errors) + prioritized_replay_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)

            if t > learning_starts and t % target_network_update_freq == 0:
                # Update target network periodically.
                update_target()

            mean_50ep_reward = round(np.mean(episode_rewards[-51:-1]), 1)
            mean_50ep_len = round(np.mean(ep_len[-51:-1]), 1)
            mean_10ep_reward = round(np.mean(episode_rewards[-11:-1]), 1)
            num_episodes = len(episode_rewards)
            if done and print_freq is not None and len(episode_rewards) % print_freq == 0:      
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("mean 50 episode reward", mean_50ep_reward)
                logger.record_tabular("mean 50 episode length", mean_50ep_len)
                logger.record_tabular("mean 10 episode reward", mean_10ep_reward)
                min_50ep_len = np.min(ep_len[-51:-1])
                logger.record_tabular("min 50 episode length", min_50ep_len)                
                logger.record_tabular("TD Error Mean", np.mean(errors))
                logger.record_tabular("TD Error Max", np.max(errors))
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()

            if (checkpoint_freq is not None and t > learning_starts and
                    num_episodes > 50 and len(episode_rewards) % (3 * print_freq) == 0):
                if saved_mean_reward is None or mean_10ep_reward > saved_mean_reward:
                    print('save')
                    if print_freq is not None:
                        logger.log("Saving model due to mean reward increase: {} -> {}".format(
                                   saved_mean_reward, mean_10ep_reward))
                    U.save_state(model_file)
                    model_saved = True
                    saved_mean_reward = mean_10ep_reward
        if model_saved:
            if print_freq is not None:
                logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
            U.load_state(model_file)

    print('Finish Training')
    return ActWrapper(act, act_params)

def get_percentile(data):
    ret = []
    for i in range(10):
        ret.append(np.percentile(data, 10 * i + 5))

    return ret