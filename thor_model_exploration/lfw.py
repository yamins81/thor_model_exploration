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
from thoreano.slm import (TheanoExtractedFeatures,
                          use_memmap)
from thoreano.classifier import (evaluate,
                                 train_asgd)

import comparisons as comp_module

DEFAULT_COMPARISONS = ['mult', 'sqrtabsdiff']

##################################
########lfw task evaluation
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
    
    preproc = configs[0].get('preproc')
    if preproc is None:
        preproc = {'global_normalize': True, 
                   'size': (200, 200)}
    X, y, Xr = get_relevant_images(dataset,
                                   preproc=preproc,
                                   splits = all_splits,
                                   dtype='float32')
    batchsize = 4
    performance_comp = {}
    feature_file_names = ['features_' + c_hash + '_' + str(i) +  '.dat' for i in range(len(configs))]
    train_pairs_filename = 'train_pairs_' + c_hash + '.dat'
    validate_pairs_filename = 'validate_pairs_' + c_hash + '.dat'
    test_pairs_filename = 'test_pairs_' + c_hash + '.dat'
    with TheanoExtractedFeatures(X, batchsize, configs, feature_file_names, tlimit=tlimit) as features_fps:
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
                                model, data = train_asgd(train_Xy, validate_Xy)
                                print('earlystopper', earlystopper.best_y)
                                result = evaluate(model, test_Xy, data)
                                loss = 1 - result['test_accuracy']/100.
                                perf.append(loss)
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
                            model, data = train_asgd(train_Xy, test_Xy)
                            loss = 1 - data['test_accuracy']/100.
                            perf.append(loss)
                            n_test_examples = len(test_Xy[0])
                            data['split'] = tts
                            data.pop('train_mean')
                            data.pop('train_std')
                            data.pop('trace')
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

def get_relevant_images(dataset, preproc, splits=None, dtype='uint8'):
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
                    shape=tuple(preproc.get('size', (200, 200))), 
                    dtype=dtype,
                    normalize=preproc.get('global_normalize', True)),
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
        if set(labels) == set([0, 1]):
            labels = 2*labels - 1
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
    """
    """
    def __init__(self, shape=None, ndim=None, dtype='float32', mode=None,
                 normalize=True):
        assert dtype == 'float32'
        self._shape = shape
        if ndim is None:
            self._ndim = None if (shape is None) else len(shape)
        else:
            self._ndim = ndim
        self._dtype = dtype
        self.mode = mode
        self.normalize = normalize

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
        imsize = self._shape[:2]
        if imsize != im.size:
            im = im.resize(imsize, Image.ANTIALIAS)
        rval = np.asarray(im, 'float32')
        if self.normalize:
            rval -= rval.mean()
            rval /= max(rval.std(), 1e-3)
        else:
            rval /= 255.0
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
