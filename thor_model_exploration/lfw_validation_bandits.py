"""
# MUST be run with develop branch of thoreano
"""
import copy

import numpy as np
import hyperopt.genson_bandits as gb
import hyperopt.genson_helpers as gh

import lfw
import model_exploration_params as params


class FG11TopBandit(gb.GensonBandit):
    param_gen = params.fg11_top_params

    def __init__(self):
        super(FG11TopBandit, self).__init__(source_string=gh.string(self.param_gen))

    @classmethod
    def evaluate(cls, config, ctrl):
        return cls.performance_func(config, ctrl)
        
    @classmethod
    def performance_func(cls, config, ctrl):
        return lfw.get_performance(None, config)


class CVPRTopBandit(FG11TopBandit):
    param_gen = params.cvpr_top_params

