"""
# MUST be run with feature/separate_activation branch of thoreano
"""

import numpy as np
import hyperopt.genson_bandits as gb
import hyperopt.genson_helpers as gh
import pymongo as pm

import lfw
import model_exploration_params as params

class LFWBandit(gb.GensonBandit):
    def __init__(self):
        super(LFWBandit, self).__init__(source_string=gh.string(params.original_params))

    @classmethod
    def evaluate(cls, config, ctrl):
        result = lfw.get_performance(None, config)
        return result


class LFWBanditModelExploration(gb.GensonBandit):
    def __init__(self):
        super(LFWBanditModelExploration, self).__init__(source_string=gh.string(params.order_value_params))

    @classmethod
    def evaluate(cls, config, ctrl, use_theano=True):
        config = params.get_reordered_model_config(config)
        result = lfw.get_performance(None, config)
        return result

