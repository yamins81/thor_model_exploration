import copy
import itertools
from hyperopt.genson_helpers import (null,
                         false,
                         true,
                         choice,
                         uniform,
                         gaussian,
                         lognormal,
                         qlognormal,
                         ref)


lnorm = {'kwargs':{'inker_shape' : choice([(3,3),(5,5),(7,7),(9,9)]),
         'outker_shape' : ref('this','inker_shape'),
         'remove_mean' : choice([0,1]),
         'stretch' : uniform(0,10),
         'threshold' : choice([null, uniform(0,10)])
         }}

lpool = {'kwargs': {'ker_shape' : choice([(3,3),(5,5),(7,7),(9,9)]),
          'order' : choice([1, 2, 10])
         }}

rescale = {'kwargs': {'stride' : 2}}

activ =  {'kwargs': {'min_out' : choice([null, 0]),
                     'max_out' : choice([1, null])}}

filter1 = dict(
         initialize=dict(
            filter_shape=choice([(3,3),(5,5),(7,7),(9,9)]),
            n_filters=choice([16, 32, 64]),
            generate=(
                'random:uniform',
                {'rseed': choice(range(5))})),
         kwargs={})

filter2 = copy.deepcopy(filter1)
filter2['initialize']['n_filters'] = choice([16, 32, 64, 128])
filter2['initialize']['generate'] = ('random:uniform', {'rseed': choice(range(5,10))})

filter3 = copy.deepcopy(filter1)
filter3['initialize']['n_filters'] = choice([16, 32, 64, 128, 256])
filter3['initialize']['generate'] = ('random:uniform', {'rseed': choice(range(10,15))})


#################
###original model
original_params = {'desc': [[('lnorm', lnorm)],
            [('fbcorr', filter1),
             ('activ', activ),
             ('lpool', lpool),
             ('rescale', rescale),
             ('lnorm', lnorm)],
            [('fbcorr', filter2),
             ('activ', activ),
             ('lpool', lpool),
             ('rescale', rescale),
             ('lnorm', lnorm)],
            [('fbcorr', filter3),
             ('activ', activ),
             ('lpool', lpool),
             ('rescale', rescale),
             ('lnorm', lnorm)],
           ]}


####################
####reordered models
order_choices = [[list(_o[:_ind]),list(_o[_ind:])] for _o in list(itertools.permutations(['lpool','activ','lnorm'])) for _ind in range(4)]
orders = choice(order_choices)

values = [{'lnorm':lnorm, 'lpool':lpool, 'activ':activ, 'filter':filter1},
          {'lnorm':lnorm, 'lpool':lpool, 'activ':activ, 'filter':filter2},
          {'lnorm':lnorm, 'lpool':lpool, 'activ':activ, 'filter':filter3}]
value_params = {'values':values}
order_value_params = {'order':orders, 'values':values}
values2 = [{'lnorm':lnorm, 'lpool':lpool, 'activ':activ, 'filter':filter1},
           {'lnorm':lnorm, 'lpool':lpool, 'activ':activ, 'filter':filter2}]
order_value_params2 = {'order':orders, 'values':values2}
                      
basic = ['lpool','activ','lnorm']
order_choices_single_removals_base = [[[list(_o[:_ind]),list(_o[_ind:])] for _o in list(itertools.permutations(_sset)) for _ind in range(len(_sset)+1)] for _sset in itertools.combinations(basic, 2)]
order_choices_single_removals = list(itertools.chain(*order_choices_single_removals_base))
order_choices_double_removals_base = [[[list(_o[:_ind]),list(_o[_ind:])] for _o in list(itertools.permutations(_sset)) for _ind in range(len(_sset)+1)] for _sset in itertools.combinations(basic, 1)]
order_choices_double_removals = list(itertools.chain(*order_choices_double_removals_base))
order_choices_removals = order_choices_single_removals + order_choices_double_removals

def get_relevant_values(_v, _ord):
    ops = _ord[0] + _ord[1] + ['filter']
    _v = copy.deepcopy(_v)
    for (_ind, _vd) in enumerate(_v):
        _v[_ind] = dict([(key,val) for (key,val) in _v[_ind].items() if key in ops])
    return _v
        
order_choices_removals_values = [{'order':_o, 'values': get_relevant_values(values, _o)} for _o in order_choices_removals]
order_removals_value_params = {'desc': choice(order_choices_removals_values)}

def get_reordered_model_config(config):
    if 'desc' in config:
        config = config['desc']
    before, after = config['order']
    values = config['values']
    layers = []
    config = {'desc':layers}
    for (layer_ind, vals) in enumerate(values):
        B = [(b, vals[b]) for b in before]
        A = [(a, vals[a]) for a in after]
        layer = B + [('fbcorr',vals['filter'])] + A + [('rescale',rescale)]
        layers.append(layer)
    return config


def remove(s,s0):
    s = s[:]
    for s0e in s0:
        s.remove(s0e)
    return s
standard_order = ['activ', 'lpool', 'lnorm']
standard_removed_orders_first = [([[], remove(standard_order,k)], [[],standard_order]) for k in itertools.chain(itertools.combinations(standard_order, 2), itertools.combinations(standard_order, 1))]
standard_removed_orders_last = [([[],standard_order], [[], remove(standard_order,k)]) for k in itertools.chain(itertools.combinations(standard_order, 2), itertools.combinations(standard_order, 1))]
standard_removed_orders_first_different = standard_removed_orders_first + standard_removed_orders_last

def get_relevant_values_first_different(_v, _ord):
    _v = copy.deepcopy(_v)
    for (_ind, _vd) in enumerate(_v):
        o_ind = 0 if _ind == 0 else 1
        ops = _ord[o_ind][0] + _ord[o_ind][1] + ['filter']
        _v[_ind] = dict([(key,val) for (key,val) in _v[_ind].items() if key in ops])
    return _v
    
standard_removed_orders_first_different_values = [{'order':_o, 'values': get_relevant_values_first_different(values,_o)} for _o in standard_removed_orders_first_different]
standard_order_removals_first_different_value_params = {'desc': choice(standard_removed_orders_first_different_values)}


def get_reordered_model_config_first_different(config):
    if 'desc' in config:
        config = config['desc']
    values = config['values']
    layers = []
    newconfig = {'desc':layers}
    for (layer_ind, vals) in enumerate(values):
        before, after = config['order'][0 if layer_ind == 0 else 1]
        B = [(b, vals[b]) for b in before]
        A = [(a, vals[a]) for a in after]
        layer = B + [('fbcorr',vals['filter'])] + A + [('rescale',rescale)]
        layers.append(layer)
    return newconfig
