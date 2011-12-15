"""
# MUST be run with feature/separate_activation branch of thoreano
"""
import copy
import itertools
import sys
import time
import os
import tempfile
import os.path as path
import hashlib

import Image
import skdata.larray
import skdata.utils
import numpy as np
import hyperopt.genson_bandits as gb
import hyperopt.genson_helpers as gh
import pymongo as pm
from hyperopt.genson_helpers import (null,
                         false,
                         true,
                         choice,
                         uniform,
                         gaussian,
                         lognormal,
                         qlognormal,
                         ref)
from thoreano.slm import (TheanoExtractedFeatures,
                          use_memmap)
from thoreano.classifier import (evaluate_classifier_normalize,
                                 train_asgd_classifier_normalize)


import comparisons as comp_module

DEFAULT_COMPARISONS = ['mult', 'sqrtabsdiff']

############
######params
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

class LFWBandit(gb.GensonBandit):
    def __init__(self):
        super(LFWBandit, self).__init__(source_string=gh.string(original_params))

    @classmethod
    def evaluate(cls, config, ctrl):
        result = get_performance(None, config)
        return result


####################
####reordered models
order_choices = [[list(o[:ind]),list(o[ind:])] for o in list(itertools.permutations(['lpool','activ','lnorm'])) for ind in range(4)]
orders = choice(order_choices)
values = [{'lnorm':lnorm, 'lpool':lpool, 'activ':activ, 'filter':filter1},
          {'lnorm':lnorm, 'lpool':lpool, 'activ':activ, 'filter':filter2},
          {'lnorm':lnorm, 'lpool':lpool, 'activ':activ, 'filter':filter3}]
order_value_params = {'order':orders, 'values':values}

def get_model_config(config):
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

class LFWBanditModelExploration(gb.GensonBandit):
    def __init__(self):
        super(LFWBanditModelExploration, self).__init__(source_string=gh.string(order_value_params))

    @classmethod
    def evaluate(cls, config, ctrl, use_theano=True):
        config = get_model_config(config)
        result = get_performance(None, config)
        return result

def make_plot():
    conn = pm.Connection()
    db = conn['hyperopt']
    Jobs = db['jobs']
    exp_key0 = 'lfw_model_exploration.LFWBandit/hyperopt.theano_bandit_algos.TheanoRandom'
    H0 = 1-np.array([x['result']['loss'] for x in Jobs.find({'exp_key': exp_key0, 'state':2})])
    NF0 =  np.array([x['result']['data']['mult'][0]['n_features'] for x in Jobs.find({'exp_key': exp_key0, 'state':2})])

    exp_key = 'lfw_model_exploration.LFWBanditModelExploration/hyperopt.theano_bandit_algos.TheanoRandom'
    Qs = [{'exp_key':exp_key,'state':2,'spec.order':o} for o in order_choices]

    L = [1-np.array([x['result']['loss'] for x in Jobs.find(q,fields=['result.loss'])]) for q in Qs]
    NF = [np.array([x['result']['data']['mult'][0]['n_features'] for x in Jobs.find(q,fields=['result.data.mult'])]) for q in Qs]

    import matplotlib.pyplot as plt
    plt.figure()
    plt.boxplot([l-H0.mean() for l in L])
    plt.plot(range(1,len(L)+1),[np.mean(l) - H0.mean() for l in L],color='green')
    plt.scatter(range(1,len(L)+1),[np.mean(l) - H0.mean() for l in L])
    od = {'lpool': 'p', 'activ': 'a', 'lnorm': 'n'}
    order_labels = [','.join([od[b] for b in Before]) + '|' + ','.join([od[b] for b in After]) for (Before, After) in order_choices]
    plt.xticks(range(1,len(L)+1),order_labels, rotation=60)
    plt.title('Model form exploration')
    plt.ylabel('Performance relative to usual L3 on LFW')
    plt.xlabel('Architecture tag')
    plt.savefig('model_exploration_boxplots.png')

    return L, (H0, NF0)


##################################
######## basic lfw task evaluation
def test_splits():
    T = ['fold_' + str(i) for i in range(10)]
    splits = []
    for i in range(10):
        inds = range(10)
        inds.remove(i)
        v_ind = (i+1) % 10
        inds.remove(v_ind)
        test = T[i]
        validate = T[v_ind]
        train = [T[ind] for ind in inds]
        splits.append({'train': train,
                       'validate': validate,
                       'test': test})
    return splits


def get_test_performance(outfile, config, use_theano=True, flip_lr=False, comparisons=DEFAULT_COMPARISONS):
    """adapter to construct split notation for 10-fold split and call
    get_performance on it (e.g. this is like "test a config on View 2")
    """
    splits = test_splits()
    return get_performance(outfile, config, train_test_splits=splits,
                           use_theano=use_theano, flip_lr=flip_lr, tlimit=None,
                           comparisons=comparisons)


def get_performance(outfile, configs, train_test_splits=None, use_theano=True,
                    flip_lr=False, tlimit=35, comparisons=DEFAULT_COMPARISONS):
    """Given a config and list of splits, test config on those splits.

    Splits can either be "view1-like", e.g. train then test or "view2-like", e.g.
    train/validate and then test.  splits specified by a list of dictionaries
    with keys in ["train","validate","test"] and values are split names recognized
    by skdata.lfw.Aligned.raw_verification_task

    This function both extracts features AND runs SVM evaluation. See functions
    "get_features" and "train_feature"  below that split these two things up.
    At some point this function should be simplified by calls to those two.
    """
    import skdata.lfw
    c_hash = get_config_string(configs)
    if isinstance(configs, dict):
        configs = [configs]
    assert all([hasattr(comp_module,comparison) for comparison in comparisons])
    dataset = skdata.lfw.Aligned()
    if train_test_splits is None:
        train_test_splits = [{'train': 'DevTrain', 'test': 'DevTest'}]
    train_splits = [tts['train'] for tts in train_test_splits]
    validate_splits = [tts.get('validate',[]) for tts in train_test_splits]
    test_splits = [tts['test'] for tts in train_test_splits]
    all_splits = test_splits + validate_splits + train_splits
    X, y, Xr = get_relevant_images(dataset, splits = all_splits, dtype='float32')
    batchsize = 4
    performance_comp = {}
    feature_file_names = ['features_' + c_hash + '_' + str(i) +  '.dat' for i in range(len(configs))]
    train_pairs_filename = 'train_pairs_' + c_hash + '.dat'
    validate_pairs_filename = 'validate_pairs_' + c_hash + '.dat'
    test_pairs_filename = 'test_pairs_' + c_hash + '.dat'
    with TheanoExtractedFeatures(X, batchsize, configs, feature_file_names,
                                 use_theano=use_theano, tlimit=tlimit) as features_fps:

        feature_shps = [features_fp.shape for features_fp in features_fps]
        datas = {}
        for comparison in comparisons:
            print('Doing comparison %s' % comparison)
            perf = []
            datas[comparison] = []
            comparison_obj = getattr(comp_module,comparison)
            #how does tricks interact with n_features, if at all?
            n_features = sum([comparison_obj.get_num_features(f_shp) for f_shp in feature_shps])
            f_info = {'feature_shapes': feature_shps, 'n_features': n_features}
            for tts in train_test_splits:
                print('Split', tts)
                if tts.get('validate') is not None:
                    train_split = tts['train']
                    validate_split = tts['validate']
                    test_split = tts['test']
                    with PairFeatures(dataset, train_split, Xr,
                            n_features, features_fps, comparison_obj,
                                      train_pairs_filename, flip_lr=flip_lr) as train_Xy:
                        with PairFeatures(dataset, validate_split,
                                Xr, n_features, features_fps, comparison_obj,
                                          validate_pairs_filename) as validate_Xy:
                            with PairFeatures(dataset, test_split,
                                Xr, n_features, features_fps, comparison_obj,
                                          test_pairs_filename) as test_Xy:
                                model, earlystopper, data, train_mean, train_std = \
                                                 train_asgd_classifier_normalize(train_Xy, validate_Xy)
                                print('earlystopper', earlystopper.best_y)
                                result = evaluate_classifier_normalize(model, test_Xy, train_mean, train_std)
                                perf.append(result['loss'])
                                print ('Split',tts, 'comparison', comparison, 'loss is', result['loss'])
                                n_test_examples = len(test_Xy[0])
                                result['split'] = tts
                                result.update(f_info)
                                datas[comparison].append(result)

                else:
                    train_split = tts['train']
                    test_split = tts['test']
                    with PairFeatures(dataset, train_split, Xr,
                            n_features, features_fps, comparison_obj,
                                      train_pairs_filename, flip_lr=flip_lr) as train_Xy:
                        with PairFeatures(dataset, test_split,
                                Xr, n_features, features_fps, comparison_obj,
                                          test_pairs_filename) as test_Xy:
                            model, earlystopper, data, train_mean, train_std = \
                                                 train_asgd_classifier_normalize(train_Xy, test_Xy)
                            perf.append(data['loss'])
                            n_test_examples = len(test_Xy[0])
                            data['split'] = tts
                            data.update(f_info)
                            datas[comparison].append(data)

            performance_comp[comparison] = float(np.array(perf).mean())

    performance = float(np.array(performance_comp.values()).min())
    result = dict(
            loss=performance,
            loss_variance=performance * (1 - performance) / n_test_examples,
            performances=performance_comp,
            data=datas,
            status='ok')

    if outfile is not None:
        outfh = open(outfile,'w')
        cPickle.dump(result, outfh)
        outfh.close()
    return result

def get_relevant_images(dataset, splits=None, dtype='uint8'):
    # load & resize logic is LFW Aligned -specific
    assert 'Aligned' in str(dataset.__class__)


    Xr, yr = dataset.raw_classification_task()
    Xr = np.array(Xr)

    if splits is not None:
        splits = unroll(splits)

    if splits is not None:
        all_images = []
        for s in splits:
            if s.startswith('re'):
                A, B, c = dataset.raw_verification_task_resplit(split=s[2:])
            else:
                A, B, c = dataset.raw_verification_task(split=s)
            all_images.extend([A,B])
        all_images = np.unique(np.concatenate(all_images))

        inds = np.searchsorted(Xr, all_images)
        Xr = Xr[inds]
        yr = yr[inds]

    X = skdata.larray.lmap(
                ImgLoaderResizer(
                    shape=(200, 200),  # lfw-specific
                    dtype=dtype),
                Xr)

    Xr = np.array([os.path.split(x)[-1] for x in Xr])

    return X, yr, Xr


class PairFeatures(object):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def work(self, dset, split, X, n_features,
             features_fps, comparison_obj, filename, flip_lr=False):
        if isinstance(split, str):
            split = [split]
        A = []
        B = []
        labels = []
        for s in split:
            if s.startswith('re'):
                s = s[2:]
                A0, B0, labels0 = dset.raw_verification_task_resplit(split=s)
            else:
                A0, B0, labels0 = dset.raw_verification_task(split=s)
            A.extend(A0)
            B.extend(B0)
            labels.extend(labels0)
        Ar = np.array([os.path.split(ar)[-1] for ar in A])
        Br = np.array([os.path.split(br)[-1] for br in B])
        labels = np.array(labels)
        Aind = np.searchsorted(X, Ar)
        Bind = np.searchsorted(X, Br)
        assert len(Aind) == len(Bind)
        pair_shp = (len(labels), n_features)


        if flip_lr:
            pair_shp = (4 * pair_shp[0], pair_shp[1])

        size = 4 * np.prod(pair_shp)
        print('Total size: %i bytes (%.2f GB)' % (size, size / float(1e9)))
        memmap = filename is not None and use_memmap(size)
        if memmap:
            print('get_pair_fp memmap %s for features of shape %s' % (
                                                    filename, str(pair_shp)))
            feature_pairs_fp = np.memmap(filename,
                                    dtype='float32',
                                    mode='w+',
                                    shape=pair_shp)
        else:
            print('using memory for features of shape %s' % str(pair_shp))
            feature_pairs_fp = np.empty(pair_shp, dtype='float32')
        feature_labels = []

        for (ind,(ai, bi)) in enumerate(zip(Aind, Bind)):
            # -- this flattens 3D features to 1D features
            if flip_lr:
                feature_pairs_fp[4 * ind + 0] = np.concatenate(
                        [comparison_obj(
                            fp[ai, :, :, :],
                            fp[bi, :, :, :])
                            for fp in features_fps])
                feature_pairs_fp[4 * ind + 1] = np.concatenate(
                        [comparison_obj(
                            fp[ai, :, ::-1, :],
                            fp[bi, :, :, :])
                            for fp in features_fps])
                feature_pairs_fp[4 * ind + 2] = np.concatenate(
                        [comparison_obj(
                            fp[ai, :, :, :],
                            fp[bi, :, ::-1, :])
                            for fp in features_fps])
                feature_pairs_fp[4 * ind + 3] = np.concatenate(
                        [comparison_obj(
                            fp[ai, :, ::-1, :],
                            fp[bi, :, ::-1, :])
                            for fp in features_fps])

                feature_labels.extend([labels[ind]] * 4)
            else:
                feats = [comparison_obj(fp[ai],fp[bi])
                        for fp in features_fps]
                feature_pairs_fp[ind] = np.concatenate(feats)
                feature_labels.append(labels[ind])
            if ind % 100 == 0:
                print('get_pair_fp  %i / %i' % (ind, len(Aind)))

        if memmap:
            print ('flushing memmap')
            sys.stdout.flush()
            del feature_pairs_fp
            self.filename = filename
            self.features = np.memmap(filename,
                    dtype='float32',
                    mode='r',
                    shape=pair_shp)
        else:
            self.features = feature_pairs_fp
            self.filename = ''

        self.labels = np.array(feature_labels)


    def __enter__(self):
        self.work(*self.args, **self.kwargs)
        return (self.features, self.labels)

    def __exit__(self, *args):
        if self.filename:
            os.remove(self.filename)


class ImgLoaderResizer(object):
    """ Load 250x250 greyscale images, return normalized 200x200 float32 ones.
    """
    def __init__(self, shape=None, ndim=None, dtype='float32', mode=None):
        assert shape == (200, 200)
        assert dtype == 'float32'
        self._shape = shape
        if ndim is None:
            self._ndim = None if (shape is None) else len(shape)
        else:
            self._ndim = ndim
        self._dtype = dtype
        self.mode = mode

    def rval_getattr(self, attr, objs):
        if attr == 'shape' and self._shape is not None:
            return self._shape
        if attr == 'ndim' and self._ndim is not None:
            return self._ndim
        if attr == 'dtype':
            return self._dtype
        raise AttributeError(attr)

    def __call__(self, file_path):
        im = Image.open(file_path)
        im = im.resize((200, 200), Image.ANTIALIAS)
        rval = np.asarray(im, 'float32')
        rval -= rval.mean()
        rval /= max(rval.std(), 1e-3)
        assert rval.shape == (200, 200)
        return rval


def unroll(X):
    Y = []
    for x in X:
        if isinstance(x,str):
            Y.append(x)
        else:
            Y.extend(x)
    return np.unique(Y)


def get_config_string(configs):
    return hashlib.sha1(repr(configs)).hexdigest()


def random_id():
    return hashlib.sha1(str(np.random.randint(10,size=(32,)))).hexdigest()
