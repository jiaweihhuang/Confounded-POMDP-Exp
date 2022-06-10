import numpy as np
import os
import pickle
import time

class ope_log_class():
    def __init__(self, path='.', name=None, tau=None, env_name=None, value_true=None):
        assert name is not None, 'log should have a name'
        self.dir_path = path
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)

        self.path = os.path.join(self.dir_path, name)
        
        self.iter = []
        self.true_rew = []
        self.bound_info = []

        assert tau is not None
        assert env_name is not None

        self.doc = {
            'Iter': self.iter,
            'tau': tau,
            'True_Rew': value_true,
            'env_name': env_name,
            'Bound_Info': self.bound_info,
        }

    def dump(self,):
        with open(self.path, 'wb') as f:
            pickle.dump(self.doc, f)

    def update_bound_info(self, iter, lower_bound, upper_bound):
        self.bound_info.append({
            'iter': iter,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
        })

    def info(self, string):
        print(string)

    def __str__(self,):
        return self.doc.__str__()