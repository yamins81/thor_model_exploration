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
value_params = {'values': values}
order_value_params = {'order':orders, 'values':values}

order_value_params_other = copy.deepcopy(order_value_params)
order_value_params_other['preproc'] = {'size': choice([
                                                       #(150, 150),
                                                       (250, 250)
                                                      ]),
                                       'global_normalize': false}


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

order_choices_removals_values_other = [{'order':_o,
                                        'values': get_relevant_values(values, _o),
                                        'preproc': {'size': choice([
                                                                     #(150, 150),
                                                                     (250, 250)
                                                                    ]),
                                                    'global_normalize': false}
                                        } for _o in order_choices_removals]
order_removals_value_params_other = {'desc': choice(order_choices_removals_values_other)}


def get_reordered_model_config(config):
    if 'desc' in config:
        config = config['desc']
    before, after = config['order']
    values = config['values']
    layers = []
    newconfig = {'desc':layers}
    for (layer_ind, vals) in enumerate(values):
        B = [(b, vals[b]) for b in before]
        A = [(a, vals[a]) for a in after]
        layer = B + [('fbcorr',vals['filter'])] + A + [('rescale',rescale)]
        layers.append(layer)
    if 'preproc' in config:
        newconfig['preproc'] = config['preproc'] 
    return newconfig


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


fg11_top_params = {'desc':[[('lnorm',{'kwargs':{'inker_shape': (9, 9),
                     'outker_shape': (9, 9),
                     'stretch':10,
                     'threshold': 1}})], 
         [('fbcorr', {'initialize': {'filter_shape': (3, 3),
                                     'n_filters': 64,
                                     'generate': ('random:uniform', 
                                                  {'rseed': choice(range(5))})},
                      'kwargs':{}}),
          ('activ', {'min_out': 0,
                     'max_out': null}),
          ('lpool', {'kwargs': {'ker_shape': (7, 7),
                                'order': 1}}),
          ('rescale', rescale),
          ('lnorm', {'kwargs': {'inker_shape': (5, 5),
                                'outker_shape': (5, 5),
                                'stretch': 0.1,
                                'threshold': 1}})],
         [('fbcorr', {'initialize': {'filter_shape': (5, 5),
                                     'n_filters': 128,
                                     'generate': ('random:uniform',
                                                  {'rseed': choice(range(5))})},
                      'kwargs': {}}),
          ('activ', {'min_out': 0,
                     'max_out': null}),
          ('lpool', {'kwargs': {'ker_shape': (5, 5),
                                'order': 1}}),
          ('rescale', rescale),  
          ('lnorm', {'kwargs': {'inker_shape': (7, 7),
                                'outker_shape': (7, 7),
                                'stretch': 1,
                                'threshold': 1}})],
         [('fbcorr', {'initialize': {'filter_shape': (5, 5),
                                     'n_filters': 256,
                                     'generate': ('random:uniform',
                                                 {'rseed': choice(range(5))})},
                      'kwargs': {}}),
           ('activ', {'min_out': 0,
                      'max_out': null}),
           ('lpool', {'kwargs': {'ker_shape': (7, 7),
                                 'order': 10}}),
           ('rescale', rescale), 
           ('lnorm', {'kwargs': {'inker_shape': (3, 3),
                                 'outker_shape': (3, 3),
                                 'stretch': 10,
                                 'threshold': 1}})]]}



cvpr_top_params = {'desc':[[('lnorm',
       {'kwargs': {'inker_shape': [5, 5],
         'outker_shape': [5, 5],
         'remove_mean': 0,
         'stretch': 0.1,
         'threshold': 1}})],
     [('fbcorr',
       {'initialize': {'filter_shape': [5, 5],
         'generate': ['random:uniform', {'rseed': 12}],
         'n_filters': 64},
        'kwargs': {}}),
      ('activ', {'kwargs': {'max_out': null, 'min_out': 0}}),  
      ('lpool', {'kwargs': {'ker_shape': [7, 7], 'order': 10}}),
      ('rescale', rescale), 
      ('lnorm',
       {'kwargs': {'inker_shape': [5, 5],
         'outker_shape': [5, 5],
         'remove_mean': 1,
         'stretch': 0.1,
         'threshold': 10}})],
     [('fbcorr',
       {'initialize': {'filter_shape': [7, 7],
         'generate': ['random:uniform', {'rseed': 24}],
         'n_filters': 64},
        'kwargs': {}}),
      ('activ', {'kwargs': {'max_out': null, 'min_out': 0}}), 
      ('lpool', {'kwargs': {'ker_shape': [3, 3], 'order': 1}}),
      ('rescale', rescale), 
      ('lnorm',
       {'kwargs': {'inker_shape': [3, 3],
         'outker_shape': [3, 3],
         'remove_mean': 1,
         'stretch': 1,
         'threshold': 0.1}})],
     [('fbcorr',
       {'initialize': {'filter_shape': [3, 3],
         'generate': ['random:uniform', {'rseed': 32}],
         'n_filters': 256},
        'kwargs': {}}),
      ('activ', {'kwargs':{'max_out': null, 'min_out': null}}),
      ('lpool', {'kwargs': {'ker_shape': [3, 3], 'order': 2}}),
      ('rescale', rescale), 
      ('lnorm',
       {'kwargs': {'inker_shape': [3, 3],
         'outker_shape': [3, 3],
         'remove_mean': 1,
         'stretch': 0.1,
         'threshold': 1}})]]}
         
def trans_fn(x):
    tdict = {'a': 'activ', 'p': 'lpool', 'n': 'lnorm'}
    ord = [[]]
    cur = 0
    for y in x:
        if y == '|':
            cur += 1
            ord.append([])
        else:
            ord[cur].append(tdict[y])
    return ord
    

ap_values = [{'lpool':lpool, 'activ':activ, 'filter':filter1},
             {'lpool':lpool, 'activ':activ, 'filter':filter2},
             {'lpool':lpool, 'activ':activ, 'filter':filter3}]
            
good_order_abbrevs = ['|ap', 'p|a', 'ap|']
good_orders = map(trans_fn, good_order_abbrevs)
good_order_value_params = {'desc': {'order': choice(good_orders), 
                                    'values': ap_values,
                                    'preproc':{'size': [200, 200],
                                    'global_normalize': choice([0, 1])}
                                    }}

good_order_abbrevs2 = ['|apn', '|anp', '|nap']
good_orders2 = map(trans_fn, good_order_abbrevs2)
good_order_value_params2 = {'desc': {'order': choice(good_orders2),
                                     'values': values,
                                     'preproc':{'size': [200, 200],
                                     'global_normalize': choice([0, 1])}
                                     }}
