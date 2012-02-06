import os
import copy

import matplotlib.pyplot as plt
import pymongo as pm
import numpy as np
import scipy.stats as stats

import model_exploration_params as params

def make_plot(save=False):
    conn = pm.Connection()
    db = conn['hyperopt']
    Jobs = db['jobs']
    exp_key0 = 'lfw_model_exploration.LFWBandit/hyperopt.theano_bandit_algos.TheanoRandom'
    H0 = 1-np.array([x['result']['loss'] for x in Jobs.find({'exp_key': exp_key0, 'state':2})])
    NF0 =  np.array([x['result']['data']['mult'][0]['n_features'] for x in Jobs.find({'exp_key': exp_key0, 'state':2})])

    exp_key = 'lfw_model_exploration.LFWBanditModelExploration/hyperopt.theano_bandit_algos.TheanoRandom'
    Qs = [{'exp_key':exp_key,'state':2,'spec.order':o} for o in params.order_choices]

    L = [1-np.array([x['result']['loss'] for x in Jobs.find(q,fields=['result.loss'])]) for q in Qs]
    NF = [np.array([x['result']['data']['mult'][0]['n_features'] for x in Jobs.find(q,fields=['result.data.mult'])]) for q in Qs]

    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.boxplot([l-H0.mean() for l in L])
    plt.plot(range(1,len(L)+1),[np.mean(l) - H0.mean() for l in L],color='green')
    plt.scatter(range(1,len(L)+1),[np.mean(l) - H0.mean() for l in L])
    od = {'lpool': 'p', 'activ': 'a', 'lnorm': 'n'}
    order_labels = [','.join([od[b] for b in Before]) + '|' + ','.join([od[b] for b in After]) for (Before, After) in params.order_choices]

    ax1 = fig.gca()
    ax2 = ax1.twinx()
    ax2.plot(range(1,len(L) + 1),[nf.mean() - NF0.mean() for nf in NF],color='r')
    ax2.set_ylabel('Mean number of Features over models (red)')

    fig.sca(ax1)
    plt.xticks(range(1,len(L)+1),order_labels, rotation=60)
    plt.suptitle('Model form exploration on LFW ',y=.95,fontsize=15)
    ax1.set_ylabel('Performance relative to usual L3 on LFW (mean is green)')
    plt.xlabel('Architecture tag')
    plt.xlim([0,len(L)+1])
    for tl in ax2.get_yticklabels():
        tl.set_color('r')
    plt.draw()

    if save:
        plt.savefig('model_exploration_boxplots_lfw.png')
    return (L, NF), (H0, NF0)




from yamutils.basic import pluck
def make_removals_plot_lfw(save=False):
    conn = pm.Connection()
    db = conn['hyperopt']
    Jobs = db['jobs']

    exp_key0 = 'lfw_model_exploration.LFWBanditModelExploration/hyperopt.theano_bandit_algos.TheanoRandom'
    H0 = Jobs.group(['spec.order'],
                   {'exp_key': exp_key0, 'state':2},
                   {'losses': []},
                   'function(d, o){o.losses.push(d.result.loss);}')

    order_choices = params.order_choices
    ords0 = pluck(H0, 'spec.order')
    reinds = [ords0.index(_o) for _o in order_choices]
    H0 = [H0[_r] for _r in reinds]
    
    exp_key = 'thor_model_exploration.model_exploration_bandits.LFWBanditRemovalsExploration/hyperopt.theano_bandit_algos.TheanoRandom'

    H = Jobs.group(['spec.desc.order'],
                   {'exp_key': exp_key, 'state':2},
                   {'losses': []},
                   'function(d, o){o.losses.push(d.result.loss);}')
        
    order_choices_removals = params.order_choices_removals
    ords = pluck(H, 'spec.desc.order')
    reinds = [ords.index(_o) for _o in order_choices_removals]
    H = [H[_r] for _r in reinds]
    
    exp_key = 'thor_model_exploration.model_exploration_bandits.LFWBanditRemovalsEvaluation/hyperopt.theano_bandit_algos.TheanoRandom'
    for (ind,l) in enumerate(Jobs.find({'exp_key': exp_key, 'state': 2}, fields=['result.order_results'])):
        print(ind)
        for _ind in range(len(l['result']['order_results'])):
            y = l['result']['order_results'][_ind]['loss']
            H[_ind]['losses'].append(y)
                   
    od = {'lpool': 'p', 'activ': 'a', 'lnorm': 'n'}
    order_labels0 = [','.join([od[b] for b in Before]) + '|' + ','.join([od[b] for b in After]) for (Before, After) in order_choices]
    order_labels = [','.join([od[b] for b in Before]) + '|' + ','.join([od[b] for b in After]) for (Before, After) in order_choices_removals]

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(18,8))
    plt.boxplot([1-np.array(h['losses']) for h in H0 + H])
    means = [1-np.array(h['losses']).mean() for h in H0 + H]
    plt.plot(range(1,len(H0)+len(H)+1), means, color='green')
    plt.scatter(range(1,len(H0)+len(H)+1), means)
    
    plt.xticks(range(1,len(ords0 + ords)+1), order_labels0 + order_labels, rotation=60)
    plt.axvline(len(ords0) + .5, linestyle='--', color='green', linewidth=2)
    plt.axvline(len(ords0) + len(params.order_choices_single_removals) + .5, linestyle='--', color='green', linewidth=2)
    
    plt.title('Model form removals on LFW Verification Task', y=.95, fontsize=15)
    plt.ylabel('Absolute performance')
    plt.xlabel('Architecture tag')
    if save:
        plt.savefig('model_exploration_removal_boxplots_lfw.png')
    

def make_standardfirstdifferent_removals_plot_lfw(save=False):
    conn = pm.Connection()
    db = conn['hyperopt']
    Jobs = db['jobs']


    exp_key0 = 'lfw_model_exploration.LFWBanditModelExploration/hyperopt.theano_bandit_algos.TheanoRandom'
    H0 = Jobs.group(['spec.order'],
                   {'exp_key': exp_key0, 'state':2, 'spec.order': [[], ['activ','lpool','lnorm']]},
                   {'losses': []},
                   'function(d, o){o.losses.push(d.result.loss);}')
                   
    order_choices = [[[], ['activ','lpool','lnorm']]]

    exp_key = 'thor_model_exploration.model_exploration_bandits.LFWBanditStandardFirstDifferentRemovalsExploration/hyperopt.theano_bandit_algos.TheanoRandom'

    H = Jobs.group(['spec.desc.order'],
                   {'exp_key': exp_key, 'state':2},
                   {'losses': []},
                   'function(d, o){o.losses.push(d.result.loss);}')
        
    standard_removed_orders_first_different = params.standard_removed_orders_first_different
    standard_removed_orders_first_different = map(list, standard_removed_orders_first_different)
    inds = [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11]
    standard_removed_orders_first_different = [standard_removed_orders_first_different[ind] for ind in inds]
    
    ords = pluck(H, 'spec.desc.order')
    reinds = [ords.index(_o) for _o in standard_removed_orders_first_different]
    H = [H[_r] for _r in reinds]

    od = {'lpool': 'p', 'activ': 'a', 'lnorm': 'n'}
    order_labels0 = [','.join([od[b] for b in After]) for (Before, After) in order_choices]
    order_labels = [','.join([od[b] for b in A]) + '*' +  ','.join([od[b] for b in C])for [[B,A],[D,C]] in standard_removed_orders_first_different]
 

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(18,8))
    plt.boxplot([1-np.array(h['losses']) for h in H0 + H])
    means = [1-np.array(h['losses']).mean() for h in H0 + H]
    plt.plot(range(1,len(H0)+len(H)+1), means, color='green')
    plt.scatter(range(1,len(H0)+len(H)+1), means)
    plt.xticks(range(1,len(order_labels0) + len(order_labels) + 1), order_labels0 + order_labels, rotation=60)
    plt.axvline(len(order_labels0) + .5, linestyle='--', color='green', linewidth=2)


    plt.title('Model form first-different removals on LFW Verification Task', y=.95, fontsize=15)
    plt.ylabel('Absolute performance')
    plt.xlabel('Architecture tag')
    if save:
        plt.savefig('model_exploration_firstdifferent_removal_boxplots_lfw.png')


def make_plot_lfw_reorder_other(save=False):
    conn = pm.Connection()
    db = conn['hyperopt']
    Jobs = db['jobs']
    
    exp_key = 'thor_model_exploration.model_exploration_bandits.LFWBanditModelExplorationOther/hyperopt.Random'

    H = Jobs.group(['spec.order'],
                   {'exp_key': exp_key, 'state':2, 
                    'spec.preproc.size.0':250
                   },
                   {'losses': []},
                   'function(d, o){o.losses.push(d.result.loss);}')
        
    order_choices = params.order_choices
    ords = pluck(H, 'spec.order')
    reinds = [ords.index(_o) for _o in order_choices]
    H = [H[_r] for _r in reinds]

    od = {'lpool': 'p', 'activ': 'a', 'lnorm': 'n'}
    order_labels = [','.join([od[b] for b in Before]) + '|' + ','.join([od[b] for b in After]) for (Before, After) in order_choices]
 
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(18,8))
    plt.boxplot([1-np.array(h['losses']) for h in H])
    means = [1-np.array(h['losses']).mean() for h in H]
    plt.plot(range(1,len(H)+1), means, color='green')
    plt.scatter(range(1,len(H)+1), means)
    
    plt.xticks(range(1,len(ords)+1),  order_labels, rotation=60)
    
    plt.ylabel('Absolute performance')
    plt.xlabel('Architecture tag')


from yamutils.basic import pluck
def make_removals_plot_lfw_other(save=False):
    conn = pm.Connection()
    db = conn['hyperopt']
    Jobs = db['jobs']

    exp_key0 = 'thor_model_exploration.model_exploration_bandits.LFWBanditModelExplorationOther/hyperopt.Random'
    H0 = Jobs.group(['spec.order'],
                   {'exp_key': exp_key0, 'state':2,
                    'spec.preproc.size.0':250
                   },
                   {'losses': []},
                   'function(d, o){o.losses.push(d.result.loss);}')

    order_choices = params.order_choices
    ords0 = pluck(H0, 'spec.order')
    reinds = [ords0.index(_o) for _o in order_choices]
    H0 = [H0[_r] for _r in reinds]
    
    exp_key = 'thor_model_exploration.model_exploration_bandits.LFWBanditRemovalsExplorationOther/hyperopt.Random'

    H = Jobs.group(['spec.desc.order'],
                   {'exp_key': exp_key, 'state':2,
                    'spec.desc.preproc.size.0':250
                   },
                   {'losses': []},
                   'function(d, o){o.losses.push(d.result.loss);}')
        
    order_choices_removals = params.order_choices_removals
    ords = pluck(H, 'spec.desc.order')
    reinds = [ords.index(_o) for _o in order_choices_removals]
    H = [H[_r] for _r in reinds]
                       
    od = {'lpool': 'p', 'activ': 'a', 'lnorm': 'n'}
    order_labels0 = [','.join([od[b] for b in Before]) + '|' + ','.join([od[b] for b in After]) for (Before, After) in order_choices]
    order_labels = [','.join([od[b] for b in Before]) + '|' + ','.join([od[b] for b in After]) for (Before, After) in order_choices_removals]

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(18,8))
    plt.boxplot([1-np.array(h['losses']) for h in H0 + H])
    means = [1-np.array(h['losses']).mean() for h in H0 + H]
    plt.plot(range(1,len(H0)+len(H)+1), means, color='green')
    plt.scatter(range(1,len(H0)+len(H)+1), means)
    
    plt.xticks(range(1,len(ords0 + ords)+1), order_labels0 + order_labels, rotation=60)
    plt.axvline(len(ords0) + .5, linestyle='--', color='green', linewidth=2)
    plt.axvline(len(ords0) + len(params.order_choices_single_removals) + .5, linestyle='--', color='green', linewidth=2)
    
    plt.ylabel('Absolute performance')
    plt.xlabel('Architecture tag')


def see_results(exp_key, group_by='spec.order'):
    conn = pm.Connection()
    db = conn['hyperopt']
    Jobs = db['jobs']
    H = Jobs.group([group_by],
                   {'exp_key': exp_key, 'state':2},
                   {'C': 0, 'T': 0, 'T2': 0, 'M': 1},
                   'function(d, o){o.C += 1; o.T += d.result.loss; o.T2 += Math.pow(d.result.loss, 2); o.M = Math.min(o.M, d.result.loss)}',
                   'function(o){o.avg = o.T/o.C; o.std = Math.sqrt(o.T2/o.C - Math.pow(o.avg,2)); o.score = 1 - o.avg; o.max_score = 1 - o.M;}'
                  )

    return H

def make_plot2(save=False):
    conn = pm.Connection()
    db = conn['hyperopt']
    Jobs = db['jobs']
    exp_key0 = 'thor_model_exploration.model_exploration_bandits.SyntheticBandit/hyperopt.theano_bandit_algos.TheanoRandom/fewer_training_examples'
    H0 = 1-np.array([x['result']['loss'] for x in Jobs.find({'exp_key': exp_key0, 'state':2})])
    NF0 =  np.array([x['result']['num_features'] for x in Jobs.find({'exp_key': exp_key0, 'state':2})])

    exp_key = 'thor_model_exploration.model_exploration_bandits.SyntheticBanditModelExploration/hyperopt.theano_bandit_algos.TheanoRandom/fewer_training_examples'
    Qs = [{'exp_key':exp_key,'state':2,'spec.order':o} for o in params.order_choices]

    L = [1-np.array([x['result']['loss'] for x in Jobs.find(q,fields=['result.loss'])]) for q in Qs]
    NF = [np.array([x['result']['num_features'] for x in Jobs.find(q,fields=['result.num_features'])]) for q in Qs]

    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.boxplot([l-H0.mean() for l in L])
    plt.plot(range(1,len(L)+1),[np.mean(l) - H0.mean() for l in L],color='green')
    plt.scatter(range(1,len(L)+1),[np.mean(l) - H0.mean() for l in L])
    od = {'lpool': 'p', 'activ': 'a', 'lnorm': 'n'}
    order_labels = [','.join([od[b] for b in Before]) + '|' + ','.join([od[b] for b in After]) for (Before, After) in params.order_choices]


    ax1 = fig.gca()
    ax2 = ax1.twinx()
    ax2.plot(range(1,len(L) + 1),[nf.mean() - NF0.mean() for nf in NF],color='r')
    ax2.set_ylabel('Mean number of Features over models (red)')

    fig.sca(ax1)
    plt.xticks(range(1,len(L)+1),order_labels, rotation=60)
    plt.suptitle('Model form exploration on Synthetic 11-Way Gray Background Categorization',y=.95,fontsize=15)
    ax1.set_ylabel('Performance relative to usual L3 on Synthetic (mean is green)')
    plt.xlabel('Architecture tag')
    plt.xlim([0,len(L)+1])
    for tl in ax2.get_yticklabels():
        tl.set_color('r')
    plt.draw()

    if save:
        plt.savefig('model_exploration_boxplots_synthetic.png')
    return (L, NF), (H0, NF0)


from yamutils.basic import pluck
def make_removals_plot_synthetic(save=False):
    conn = pm.Connection()
    db = conn['hyperopt']
    Jobs = db['jobs']

    exp_key0 = 'thor_model_exploration.model_exploration_bandits.SyntheticBanditModelExploration/hyperopt.theano_bandit_algos.TheanoRandom/fewer_training_examples'
    H0 = Jobs.group(['spec.order'],
                   {'exp_key': exp_key0, 'state':2},
                   {'losses': [], 'count': 0},
                   'function(d, o){o.losses.push(d.result.loss); o.count += 1}')

    H0e = Jobs.group(['spec.order'],
                   {'exp_key': exp_key0, 'state':3},
                   {'count': 0},
                   'function(d, o){o.count += 1;}')

    order_choices = params.order_choices
    ords0 = pluck(H0, 'spec.order')
    reinds = [ords0.index(_o) for _o in order_choices]
    H0 = [H0[_r] for _r in reinds]
    ords0e = pluck(H0e, 'spec.order')
    reinds_e = [ords0e.index(_o) for _o in order_choices]
    H0e = [H0e[_r] for _r in reinds_e]    
    
    exp_key = 'thor_model_exploration.model_exploration_bandits.SyntheticBanditRemovalsExploration/hyperopt.theano_bandit_algos.TheanoRandom'

    H = Jobs.group(['spec.desc.order'],
                   {'exp_key': exp_key, 'state':2},
                   {'losses': [], 'count': 0},
                   'function(d, o){o.losses.push(d.result.loss); o.count += 1}')
    He = Jobs.group(['spec.desc.order'],
                   {'exp_key': exp_key, 'state':3},
                   {'count': 0},
                   'function(d, o){o.count += 1;}')
        
    order_choices_removals = params.order_choices_removals
    ords = pluck(H, 'spec.desc.order')
    reinds = [ords.index(_o) for _o in order_choices_removals]
    H = [H[_r] for _r in reinds]
    ordse = pluck(He, 'spec.desc.order')
    reinds_e = [ordse.index(_o) if _o in ordse else None for _o in order_choices_removals]
    He = [He[_r] if _r is not None else {'count': 0} for _r in reinds_e]    
                       
    od = {'lpool': 'p', 'activ': 'a', 'lnorm': 'n'}
    order_labels0 = [','.join([od[b] for b in Before]) + '|' + ','.join([od[b] for b in After]) + ' (' + str(he['count']) + '/' + str(h['count']) + ')' for (Before, After), he, h in zip(order_choices,H0e, H0)]
    order_labels = [','.join([od[b] for b in Before]) + '|' + ','.join([od[b] for b in After]) + ' (' + str(he['count']) + '/' + str(h['count']) + ')' for (Before, After), he, h in zip(order_choices_removals, He, H)]

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(18,8))
    plt.boxplot([1-np.array(h['losses']) for h in H0 + H])
    means = [1-np.array(h['losses']).mean() for h in H0 + H]
    plt.plot(range(1,len(H0)+len(H)+1), means, color='green')
    plt.scatter(range(1,len(H0)+len(H)+1), means)
    
    plt.xticks(range(1,len(ords0 + ords)+1), order_labels0 + order_labels, rotation=60)
    plt.axvline(len(ords0) + .5, linestyle='--', color='green', linewidth=2)
    plt.axvline(len(ords0) + len(params.order_choices_single_removals) + .5, linestyle='--', color='green', linewidth=2)
    
    plt.title('Model form removals on Synthetic Categorization Task', y=.95, fontsize=15)
    plt.ylabel('Absolute performance')
    plt.xlabel('Architecture tag')
    if save:
        plt.savefig('model_exploration_removal_boxplots_Synthetic.png')
    


def rgetattr(d,k):
    k.split = k.split('.')
    if len(k) > 1:
        return rgetattr(d[k[0]],'.'.join(k[1:]))
    else:
        return d[k[0]]

idf = lambda x: x
first = lambda x: x[0] if x is not None else x
DISCRETE_PARAMS_BASE = [('activ.kwargs.max_out', idf,'max_out'),
          ('activ.kwargs.min_out', idf,'min_out'),
          ('filter.initialize.filter_shape', first,'filter_shape'),
          ('filter.initialize.n_filters', idf,'num_filters'),
          ('lnorm.kwargs.inker_shape', first,'norm_shape'),
          ('lnorm.kwargs.remove_mean', idf,'remove_mean'),
          ('lnorm.kwargs.stretch', None, 'stretch'),
          ('lnorm.kwargs.threshold', None, 'threshold'),
          ('lpool.kwargs.ker_shape', first,'pool_shape'),
          ('lpool.kwargs.order', idf,'pool_order')]

odict = {'lpool': 'p', 'lnorm': 'n', 'activ':'a'}
ofunc = lambda x: str('|'.join([''.join([odict[yy] for yy in y]) for y in x]))
DISCRETE_PARAMS = [('order', ofunc, 'order')] + [('values.' + str(_ind) + '.' + _p, _f, _n + str(_ind+1)) for _ind in range(3) for (_p, _f, _n) in DISCRETE_PARAMS_BASE]


def nothres_ranks():
    def rank_func(exp_key):
        import matplotlib.pyplot as plt
        conn = pm.Connection()
        db = conn['hyperopt']
        Jobs = db['jobs']

        base_q = {'exp_key': exp_key, 'state':2, 'spec.values.0.lnorm.kwargs.threshold':None,
                  'spec.values.1.lnorm.kwargs.threshold':None, 'spec.values.2.lnorm.kwargs.threshold':None}
        A1 = np.array([x['result']['loss'] for x in Jobs.find(base_q).sort('result.loss')])
        base_q2 = {'exp_key': exp_key, 'state':2}
        A2 = np.array([x['result']['loss'] for x in Jobs.find(base_q2).sort('result.loss')])

        R = np.searchsorted(A2,A1)/float(len(A2))

        return R

    exp_key1 = 'thor_model_exploration.model_exploration_bandits.SyntheticBanditModelExploration/hyperopt.theano_bandit_algos.TheanoRandom/fewer_training_examples'
    R1 = rank_func(exp_key1)
    exp_key2 = 'lfw_model_exploration.LFWBanditModelExploration/hyperopt.theano_bandit_algos.TheanoRandom'
    R2 = rank_func(exp_key2)

    plt.figure()
    plt.plot(np.arange(0,1,1./len(R1)),R1)
    plt.plot(np.arange(0,1,1./len(R2)),R2)
    plt.legend(('Synthetic','LFW'), loc='upper left')
    plt.plot(np.arange(0,1,1./len(R2)), np.arange(0,1,1./len(R2)), color='black')
    plt.title('Synthetic and LFW no-threshold vs all ranks')
    plt.ylabel('Relative rank to full distribution')
    plt.xlabel('Absolute rank within no-threshold subset')
    plt.savefig('No_threshold_ranking_comparison.png')



def analysis_nothresh(exp_key, ttl, outfile):

    conn = pm.Connection()
    db = conn['hyperopt']
    Jobs = db['jobs']

    base_q = {'exp_key': exp_key, 'state':2, 'spec.values.0.lnorm.kwargs.threshold':None,
              'spec.values.1.lnorm.kwargs.threshold':None, 'spec.values.2.lnorm.kwargs.threshold':None}

    params = [('order', ofunc, 'order')] + [('values.' + str(_ind) + '.' + _p, _f, _n + str(_ind+1)) for _ind in range(3) for (_p, _f, _n) in DISCRETE_PARAMS_BASE if _n != 'threshold']
    analysis_core(base_q, outfile, ttl, params)


def analysis_synthetic_nothresh():
    exp_key = 'thor_model_exploration.model_exploration_bandits.SyntheticBanditModelExploration/hyperopt.theano_bandit_algos.TheanoRandom/fewer_training_examples'
    analysis_nothresh(exp_key, 'synthetic model exploration analysis -- no threshold', 'nothresh_analysis_synthetic.png')


def analysis_synthetic():
    exp_key = 'thor_model_exploration.model_exploration_bandits.SyntheticBanditModelExploration/hyperopt.theano_bandit_algos.TheanoRandom/fewer_training_examples'
    analysis(exp_key, 'synthetic_analysis', 'synthetic model exploration analysis')

def analysis_lfw():
    exp_key = 'lfw_model_exploration.LFWBanditModelExploration/hyperopt.theano_bandit_algos.TheanoRandom'
    analysis(exp_key, 'lfw_analysis','lfw model exploration analysis')


def analysis(exp_key, outdir, base_ttl):
    os.mkdir(outdir)
    conn = pm.Connection()
    db = conn['hyperopt']
    Jobs = db['jobs']

    base_q = {'exp_key': exp_key, 'state':2}
    analysis_core(base_q, os.path.join(outdir,'_overall.png'), base_ttl, DISCRETE_PARAMS)
    orders = params.order_choices
    for order in orders:
        qry = copy.deepcopy(base_q)
        qry['spec.order'] = order
        ttl = base_ttl + ' for model order ' + ofunc(order)
        path = os.path.join(outdir,ofunc(order) + '.png')
        analysis_core(qry, path, ttl, Jobs=Jobs)


def analysis_removals(exp_key, outdir, base_ttl):
    os.mkdir(outdir)
    conn = pm.Connection()
    db = conn['hyperopt']
    Jobs = db['jobs']

    base_q = {'exp_key': exp_key, 'state':2}
    analysis_core(base_q, os.path.join(outdir,'_overall.png'), base_ttl, DISCRETE_PARAMS, pref='spec.desc.')
    orders = params.order_choices_removals
    for order in orders:
        qry = copy.deepcopy(base_q)
        qry['spec.desc.order'] = order
        ttl = base_ttl + ' for model order ' + ofunc(order)
        path = os.path.join(outdir,ofunc(order) + '.png')
        analysis_core(qry, path, ttl, DISCRETE_PARAMS, Jobs=Jobs, pref='spec.desc.')


def before_after_lfw():
    before_after_analysis('lfw_model_exploration.LFWBanditModelExploration/hyperopt.theano_bandit_algos.TheanoRandom',
                          'LFW',
                          'LFW')


def before_after_synthetic():
    before_after_analysis('thor_model_exploration.model_exploration_bandits.SyntheticBanditModelExploration/hyperopt.theano_bandit_algos.TheanoRandom/fewer_training_examples',
                          'synthetic',
                          'Synthetic')


def before_after_analysis(exp_key, outroot, ttlroot):
    before_q = {'exp_key': exp_key, 'state':2, 'spec.order.1':[]}
    after_q = {'exp_key': exp_key, 'state':2, 'spec.order.0':[]}
    analysis_core(before_q, 'before_' + outroot + '.png', ttlroot + ' before orders')
    analysis_core(after_q, 'after_' + outroot + '.png', ttlroot + ' after orders')

def before_after_lfw2():
    before_after_analysis2('lfw_model_exploration.LFWBanditModelExploration/hyperopt.theano_bandit_algos.TheanoRandom',
                          'LFW',
                          'LFW')

def before_after_synthetic2():
    before_after_analysis2('thor_model_exploration.model_exploration_bandits.SyntheticBanditModelExploration/hyperopt.theano_bandit_algos.TheanoRandom/fewer_training_examples',
                          'synthetic',
                          'Synthetic')

def before_after_analysis2(exp_key, outroot, ttlroot):
    before_q = {'exp_key': exp_key, 'state':2, 'spec.order.1':[]}
    after_q = {'exp_key': exp_key, 'state':2, 'spec.order.0':[]}
    analysis_core([('before',before_q),('after',after_q)], 'before_after_' + outroot + '.png', ttlroot, boxplots=True, do_max = False, do_min = False)

def analysis_core(queries, outfile, ttl, params=DISCRETE_PARAMS, Jobs=None, boxplots = True, percentiles = [], do_max = True, do_min = True, pref='spec.'):
    if isinstance(queries,dict):
        queries = [('',queries)]

    if percentiles is None:
        percentiles = []

    if Jobs is None:
        conn = pm.Connection()
        db = conn['hyperopt']
        Jobs = db['jobs']

    Max = lambda _s: np.max(_s) if len(_s) > 0 else None
    Min = lambda _s: np.min(_s) if len(_s) > 0 else None
    Pct = lambda _s, _pct: stats.scoreatpercentile(_s, _pct) if len(_s) > 0 else None

    fig = plt.figure(figsize=(16,12))
    for p_ind, (param, lfunc, pname) in enumerate(params):
        p = plt.subplot(6,6,p_ind+1)
        lines = []
        for (qryname,qry) in queries:
            L = Jobs.group([pref + param],
                           qry,
                           {'A': []},
                           'function(d, o){o.A.push(d.result.loss);}',
                          )

            if lfunc is None:
                lfunc_use = idf
                ranges = [2,4,6,8,10]
                l0 = [{pref + param:r, 'A':[]} for r in ranges]
                newL = []
                for l in L:
                    if l[pref + param] is not None:
                        ind = np.searchsorted(ranges,l[pref + param])
                        l0[ind]['A'].extend(l['A'])
                    else:
                        newL.append(l)
                newL.extend(l0)
                L = newL
            else:
                lfunc_use=lfunc

            vals = np.array([lfunc_use(h[pref+param]) for h in L])

            s = vals.argsort()
            vals_sort = vals[s]

            if boxplots:
                p.boxplot([L[_i]['A'] for _i in s])

            lines.append(p.plot(range(1,len(vals_sort)+1),[np.mean(L[_i]['A']) for _i in s]))
            if do_max:
                lines.append(p.plot(range(1,len(vals_sort)+1),[Max(L[_i]['A']) for _i in s]))
            if do_min:
                lines.append(p.plot(range(1,len(vals_sort)+1),[Min(L[_i]['A']) for _i in s]))
            for pct in percentiles:
                lines.append(p.plot(range(1,len(vals_sort)+1),[Pct(L[_i]['A'], pct) for _i in s]))
        plt.xticks(range(1,len(vals_sort)+1),vals_sort,rotation=30,ha='left')
        plt.title(pname)

    linenames = []
    for x,y in queries:
        for st in ['mean'] + (['max'] if do_max else []) + (['min'] if do_min else []) + [str(y) + 'p' for y in percentiles]:
            linenames.append(x + ' ' + st)
    plt.figlegend(lines,linenames,'center left')

    plt.subplots_adjust(hspace=.4)
    plt.suptitle(ttl + ' (losses -- smaller is better!)', fontsize=20, y=.95)
    plt.draw()
    plt.savefig(outfile)
    plt.close('all')

def get_vals(vals):
    V = []
    pr = lambda x,y : x if x is not None else y
    for v in vals:
        V.extend([
         pr(v['activ']['kwargs']['max_out'],-1000),
         pr(v['activ']['kwargs']['min_out'],1000),
         v['filter']['initialize']['filter_shape'][0],
         v['filter']['initialize']['n_filters'],
         v['lnorm']['kwargs']['inker_shape'][0],
         v['lnorm']['kwargs']['remove_mean'],
         v['lnorm']['kwargs']['stretch'],
         pr(v['lnorm']['kwargs']['threshold'],0),
         v['lpool']['kwargs']['ker_shape'][0],
         v['lpool']['kwargs']['order']])
    return tuple(V)




import scikits.learn.linear_model as linear_model
def make_regression():
    conn = pm.Connection()
    db = conn['hyperopt']
    Jobs = db['jobs']
    exp_key = 'thor_model_exploration.model_exploration_bandits.SyntheticBanditModelExploration/hyperopt.theano_bandit_algos.TheanoRandom/fewer_training_examples'

    q = {'exp_key':exp_key, 'state':2}
    L = Jobs.find(q, fields=['result.loss','spec','result.num_features'])

    X = [(l['result']['loss'], params.order_choices.index(l['spec']['order']),l['result']['num_features']) + get_vals(l['spec']['values']) for l in L]
    A = np.array([x[2:] for x in X]).astype(np.float)
    y = np.array([x[1] for x in X])
    z = np.array([y == ind for ind in range(len(params.order_choices))]).T.astype(np.int)
    A = np.column_stack([A,z])
    B = np.array([x[0] for x in X])
    cls = linear_model.RidgeCV()
    print('fitting')
    cls.fit(A,B)

    N = ['max_out','min_out','filter_shape','n_filters','norm_shape','remove_mean','stretch','threshold','pool_shape','pool_order']
    names = ['num_features'] + [n + '_' + str(i) for i in range(3) for n in N] + ['order_' + str(i) for i in range(len(params.order_choices))]
    return cls, A, B, names



import scipy.stats as stats
def make_pearson():
    conn = pm.Connection()
    db = conn['hyperopt']
    Jobs = db['jobs']

    exp_key = 'thor_model_exploration.model_exploration_bandits.SyntheticBanditModelExploration/hyperopt.theano_bandit_algos.TheanoRandom/fewer_training_examples'
    exp_key1 = 'lfw_model_exploration.LFWBanditModelExploration/hyperopt.theano_bandit_algos.TheanoRandom'

    fn = lambda x : dict([(params.order_choices.index(h['spec.order']),h['avg']) for h in x])
    fn2 = lambda x : [x[ind] for ind in range(len(params.order_choices))]

    H = fn2(fn(Jobs.group(['spec.order'],
                   {'exp_key': exp_key, 'state':2},
                   {'C': 0, 'T': 0, 'T2': 0},
                   'function(d, o){o.C += 1; o.T += d.result.loss; o.T2 += Math.pow(d.result.loss, 2);}',
                   'function(o){o.avg = o.T/o.C; o.std = Math.sqrt(o.T2/o.C - Math.pow(o.avg,2)); o.score = 1 - o.avg;}'
          )))

    H1 = fn2(fn(Jobs.group(['spec.order'],
                   {'exp_key': exp_key1, 'state':2},
                   {'C': 0, 'T': 0, 'T2': 0},
                   'function(d, o){o.C += 1; o.T += d.result.loss; o.T2 += Math.pow(d.result.loss, 2);}',
                   'function(o){o.avg = o.T/o.C; o.std = Math.sqrt(o.T2/o.C - Math.pow(o.avg,2)); o.score = 1 - o.avg;}'
                  )))

    return stats.pearsonr(H, H1)


def make_hists(exp_key,outfile,ttl):
    conn = pm.Connection()
    db = conn['hyperopt']
    Jobs = db['jobs']

    Qs = [{'exp_key':exp_key,'state':2,'spec.order':o} for o in params.order_choices]
    L = [1-np.array([x['result']['loss'] for x in Jobs.find(q,fields=['result.loss'])]) for q in Qs]
    #bins = np.arange(min(map(np.min,L)),max(map(np.max,L)),.01)
    H = [np.histogram(l) for l in L]


    od = {'lpool': 'p', 'activ': 'a', 'lnorm': 'n'}
    pr = lambda x: ','.join([od[y] for y in x[0]]) + '|' + ','.join([od[y] for y in x[1]])
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(20,20))
    for ind,(h,b) in enumerate(H):
        p = plt.subplot(6,4,ind+1)
        p.bar(b[:-1],h,width=b[1]-b[0])
        p.set_title(pr(params.order_choices[ind]))
        p.set_xticks(b[::2])
        p.set_xticklabels(['%.2f' % t for t in b[::2]])
    plt.subplots_adjust(hspace=.5)
    plt.suptitle(ttl, fontsize=20)
    plt.draw()
    #exp_key1 = 'lfw_model_exploration.LFWBanditModelExploration/hyperopt.theano_bandit_algos.TheanoRandom'
    #Qs1 = [{'exp_key':exp_key1,'state':2,'spec.order':o} for o in params.order_choices]
    #L1 = [1-np.array([x['result']['loss'] for x in Jobs.find(q,fields=['result.loss'])]) for q in Qs1]
    plt.savefig(outfile)


def make_hists_synthetic():
    exp_key = 'thor_model_exploration.model_exploration_bandits.SyntheticBanditModelExploration/hyperopt.theano_bandit_algos.TheanoRandom/fewer_training_examples'
    make_hists(exp_key,'synthetic_hists.png','Synthetic hists')

def make_hists_lfw():
    exp_key = 'lfw_model_exploration.LFWBanditModelExploration/hyperopt.theano_bandit_algos.TheanoRandom'
    make_hists(exp_key,'lfw_hists.png', 'LFW hists')


