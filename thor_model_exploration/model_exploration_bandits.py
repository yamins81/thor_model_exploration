"""
# MUST be run with feature/separate_activation branch of thoreano
"""

import numpy as np
import hyperopt.genson_bandits as gb
import hyperopt.genson_helpers as gh
import pymongo as pm

import lfw
import synthetic
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


all_invariance_query = {'config.image.bg_id':'gray.tdl',
                'config.image.s': {'$exists': True},
                'config.image.ty': {'$exists': True},
                'config.image.rxy': {'$exists': True},
                'config.image.rxz': {'$exists': True},
                '__hash__': 'e02a3c3c19a43b6aae8c4e5d8d805a96c68dd82e'}


class SyntheticBandit(gb.GensonBandit):

    def __init__(self):
        super(SyntheticBandit, self).__init__(source_string=gh.string(params.original_params))

    @classmethod
    def evaluate(cls, config, ctrl):
        result = synthetic.get_performance(config, all_invariance_query)
        return result


class SyntheticBanditModelExploration(gb.GensonBandit):

    def __init__(self):
        super(SyntheticBanditModelExploration, self).__init__(source_string=gh.string(params.order_value_params))

    @classmethod
    def evaluate(cls, config, ctrl, use_theano=True):
        config = params.get_reordered_model_config(config)
        result = synthetic.get_performance(None, config, all_invariance_query)
        return result
