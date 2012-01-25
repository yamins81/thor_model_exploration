"""
# MUST be run with develop branch of thoreano
"""
import copy

import numpy as np
import hyperopt.genson_bandits as gb
import hyperopt.genson_helpers as gh
import pymongo as pm

import lfw
import synthetic
import model_exploration_params as params


#######base bandits
class BaseBandit(gb.GensonBandit):
    param_gen = params.original_params

    def __init__(self):
        super(BaseBandit, self).__init__(source_string=gh.string(self.param_gen))

    @classmethod
    def evaluate(cls, config, ctrl):
        return cls.performance_func(config)


class ModelExplorationBase(BaseBandit):
    param_gen = params.order_value_params

    @classmethod
    def config_gen_func(cls, config):
        return params.get_reordered_model_config(config)

    @classmethod
    def evaluate(cls, config, ctrl):
        desc = cls.config_gen_func(config)
        return cls.performance_func(desc, ctrl)


class ModelEvaluationBase(ModelExplorationBase):
    param_gen = params.value_params
    orders = params.order_choices

    @classmethod
    def evaluate(cls, config, ctrl):
        result = {}
        result['order_results'] = []
        for order in cls.orders:
            order_value_config = copy.deepcopy(config)
            order_value_config['order'] = order
            desc = cls.config_gen_func(order_value_config)
            try:
                ord_res = cls.performance_func(desc, ctrl)
            except:
                ord_res = {'status': 'error'}
            result['order_results'].append(ord_res)
        result['loss'] = np.mean([ord_res.get('loss', 1) for ord_res in result['order_results']])
        result['status'] = 'ok'
        return result


class RemovalsExplorationBase(ModelExplorationBase):
    param_gen = params.order_removals_value_params


class RemovalsEvaluationBase(ModelEvaluationBase):
    orders = params.order_choices_removals


class FirstDifferentBase(object):
    @classmethod
    def config_gen_func(cls, config):
        return params.get_reordered_model_config_first_different(config)


class StandardFirstDifferentRemovalsExplorationBase(FirstDifferentBase, ModelExplorationBase):
    param_gen = params.standard_order_removals_first_different_value_params


class StandardFirstDifferentRemovalsEvaluationBase(FirstDifferentBase, ModelEvaluationBase):
    orders = params.standard_removed_orders_first_different



#######LFW
class LFWBase(object):

    @classmethod
    def performance_func(cls, config, ctrl):
        return lfw.get_performance(None, config)


class LFWBandit(BaseBandit, LFWBase):
    pass


class LFWBanditModelExploration(ModelExplorationBase, LFWBase):
    pass


class LFWBanditRemovalsEvaluation(RemovalsEvaluationBase, LFWBase):
    pass


class LFWBanditRemovalsExploration(RemovalsExplorationBase, LFWBase):
    pass


class LFWBanditStandardFirstDifferentRemovalsExploration(StandardFirstDifferentRemovalsExplorationBase, LFWBase):
    pass



#######Synthetics
synthetic_invariance_query = {'config.image.bg_id':'gray.tdl',
                'config.image.s': {'$exists': True},
                'config.image.ty': {'$exists': True},
                'config.image.rxy': {'$exists': True},
                'config.image.rxz': {'$exists': True},
                '__hash__': 'e02a3c3c19a43b6aae8c4e5d8d805a96c68dd82e'}


class SyntheticBase(object):

    @classmethod
    def performance_func(cls, config, ctrl):
        return synthetic.get_performance(config, synthetic_invariance_query)


class SyntheticBandit(BaseBandit, SyntheticBase):
    pass


class SyntheticBanditModelExploration(ModelExplorationBase, SyntheticBase):
    pass


class SyntheticBanditRemovalsExploration(RemovalsExplorationBase, SyntheticBase):
    pass


class SyntheticBanditStandardFirstDifferentRemovalsExploration(StandardFirstDifferentRemovalsExplorationBase, SyntheticBase):
    pass
