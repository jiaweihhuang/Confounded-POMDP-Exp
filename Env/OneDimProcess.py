import numpy as np

class Process():
    def __init__(self, delta, obs_noise, POMDP, max_ep_len, inherent_noise):
        self.delta = delta
        self.obs_noise = obs_noise
        self.POMDP = POMDP
        self.max_ep_len = max_ep_len

        self.inherent_noise = inherent_noise

    def reset(self):
        self.state = np.random.normal(scale=0.1)
        self.observation = self.create_observation(self.state)
        self.counter = 0
        if self.POMDP:
            return self.observation
        else:
            return self.state

    def step(self, act):
        assert act in [0, 1]
        state = self.state * 0.5 + self.delta * (2. * act - 1.) + np.random.normal(scale=self.inherent_noise)
        reward = state + self.delta * (2. * act - 1.)
        self.observation = self.create_observation(state)
        self.state = state

        self.counter += 1

        if self.POMDP:
            return self.observation, reward, self.counter >= self.max_ep_len, {'state': self.state}
        else:
            return self.state, reward, self.counter >= self.max_ep_len, {'obs': self.observation}

    def get_current_state(self,):
        return self.state

    def create_observation(self, state):
        return state + np.random.normal(scale=self.obs_noise)


'''
w > 0:
    obs > 0: act = 0
    obs < 0: act = 1

    therefore, obs will tend to be around zero
    
w < 0:
    obs > 0: act = 1
    obs < 0: act = 0

    therefore, obs will tend to be +\infty or -\infty
''' 
class Sigmoid_Policy():
    def __init__(self, w, b=0.0):
        self.w = w
        self.b = b
    
    def sample_action(self, inp):
        p0 = self.sigmoid(inp)
        return np.random.choice([0, 1], p=[p0, 1.0 - p0])

    def sigmoid(self, val):
        return 1.0 / (1.0 + np.exp(-val * self.w + self.b))

    def get_prob_with_act(self, obs, act):
        p0 = self.sigmoid(obs)
        
        prob = p0 * (1.0 - act) + (1.0 - p0) * act
        return prob
