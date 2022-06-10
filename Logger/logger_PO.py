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
        self.est_value = []
        self.f_loss = []
        self.g_loss = []

        assert tau is not None
        assert env_name is not None

        self.doc = {
            'Iter': self.iter,
            'tau': tau,
            'True_Rew': value_true,
            'env_name': env_name,
            'est_value': self.est_value,
            'f_loss': self.f_loss,
            'g_loss': self.g_loss,
        }

    def dump(self,):
        with open(self.path, 'wb') as f:
            pickle.dump(self.doc, f)

    def update_info(self, iter, info):
        for k in info.keys():
            if k not in self.doc.keys():
                self.doc[k] = []
            self.doc[k].append((iter, info[k]))

    def info(self, string):
        print(string)

    def __str__(self,):
        return self.doc.__str__()
