import copy
import itertools
import sys
import time
import os
import tempfile
import os.path as path
import hashlib

import Image
import skdata.larray as larray
import skdata.utils
import numpy as np
import pymongo as pm
import gridfs
from thoreano.slm import (FeatureExtractor,
                          slm_from_config)
from thoreano.classifier import (evaluate_classifier_normalize,
                                 train_asgd_classifier_normalize,
                                 train_scikits)


##################################
#######synthetic task evaluation

class ImgLoader(object):
    def __init__(self, fs, shape=None, ndim=None, dtype='uint8', mode=None):
        self._shape = shape
        if ndim is None:
            self._ndim = None if (shape is None) else len(shape)
        else:
            self._ndim = ndim
        self._dtype = dtype
        self.mode = mode
        self.fs = fs

    def rval_getattr(self, attr, objs):
        if attr == 'shape' and self._shape is not None:
            return self._shape
        if attr == 'ndim' and self._ndim is not None:
            return self._ndim
        if attr == 'dtype':
            return self._dtype
        raise AttributeError(attr)

    def __call__(self, file_path):
        im = Image.open(self.fs.get_version(file_path))
        if im.mode != self.mode:
            im = im.convert(self.mode)
        imsize = self._shape[:2]
        if imsize != im.size:
            im = im.resize(imsize, Image.ANTIALIAS)
        rval = np.asarray(im, dtype='float32')
        rval -= rval.mean()
        rval /= max(rval.std(), 1e-6)
        return rval


class Imageset(object):
    def __init__(self, coll, fs, query):
        self.coll = coll
        self.fs = fs
        self.query = query
        cursor = coll.find(query).sort('filename')
        self.meta = list(cursor)
        self.filenames = [m['filename'] for m in self.meta]
        self.imgs = larray.lmap(ImgLoader(fs, ndim=3, shape=(200, 200, 3),  mode='RGB'),
                                self.filenames)

    def generate_splits(self, seed, ntrain, ntest, num_splits, labelset=None, catfunc=None):
        meta = self.meta
        if labelset is not None:
            assert catfunc is not None
        else:
            labelset = [0]
            catfunc = lambda x : 0

        rng = np.random.RandomState(seed)
        splits = {}
        for split_id in range(num_splits):
            splits['train_' + str(split_id)] = []
            splits['test_' + str(split_id)] = []
            for label in labelset:
                cat = [m for m in meta if catfunc(m) == label]
                L = len(cat)
                assert L >= ntrain + ntest, 'category %s too small' % name
                perm = rng.permutation(L)
                for ind in perm[:ntrain]:
                    splits['train_' + str(split_id)].append(str(cat[ind]['filename']))
                for ind in perm[ntrain: ntrain + ntest]:
                    splits['test_' + str(split_id)].append(str(cat[ind]['filename']))
        return splits


def get_features(dataset, config):
    X = dataset.imgs
    batchsize = 4
    slm = slm_from_config(config, X.shape, batchsize=batchsize)
    extractor = FeatureExtractor(X, slm, batchsize=batchsize, verbose=True)
    features = extractor.compute_features()
    return features


def traintest(dataset, features, catfunc, seed=0, ntrain=10, ntest=10, num_splits=5):
    Xr = np.array(map(str,[m['filename'] for m in dataset.meta]))
    labels = np.array([catfunc(m) for m in dataset.meta])
    labelset = sorted(list(set(labels)))
    labeldict = dict([(l,ind) for ind,l in enumerate(labelset)])
    label_ids = np.array([labeldict[l] for l in labels])
    splits = dataset.generate_splits(seed, ntrain, ntest, num_splits, labelset=labelset, catfunc=catfunc)
    results = []
    for ind in range(num_splits):
        train_split = np.array(splits['train_' + str(ind)])
        test_split = np.array(splits['test_' + str(ind)])
        train_inds = np.searchsorted(Xr,train_split)
        test_inds = np.searchsorted(Xr,test_split)
        train_X = features[train_inds]
        test_X = features[test_inds]
        train_y = label_ids[train_inds]
        test_y = label_ids[test_inds]
        train_Xy = (train_X, train_y)
        test_Xy = (test_X, test_y)
        print(len(train_y),len(test_y))
        #model, earlystopper, result = train_asgd_classifier_normalize(train_Xy, test_Xy, verbose=True)
        model, result = train_scikits(train_Xy, test_Xy, 'liblinear', regression=False)
        results.append(result)
    return results


def dict_inverse(x):
    y = {}
    for k in x.keys():
        for kk in x[k]:
            if kk in y:
                y[kk].append(k)
            else:
                y[kk] = [k]
    return y

from synthetic_model_categories import MODEL_CATEGORIES
MODEL_CATEGORIES_INVERTED = dict_inverse(MODEL_CATEGORIES)

def get_performance(config, im_query):
    """
    """
    conn = pm.Connection()
    db = conn['thor']
    
    coll = db['images.files']
    fs = gridfs.GridFS(db,'images')
    dataset = Imageset(coll, fs, im_query)

    features = get_features(dataset, config)
    fs = features.shape
    num_features = fs[1]*fs[2]*fs[3]

    record = {}
    record['num_features'] = num_features
    record['feature_shape'] = fs

    features = features.reshape((fs[0],num_features))
    STATS = ['train_accuracy','train_ap','train_auc','test_accuracy','test_ap','test_auc']
    catfunc = lambda x: MODEL_CATEGORIES_INVERTED[x['config']['image']['model_id']][0]
    results = traintest(dataset, features, catfunc)
    stats = {}
    for stat in STATS:
        stats[stat] = np.mean([r[1][stat] for r in results])
    record['training_data'] =  stats

    record['loss'] = 1 - (record['training_data']['test_accuracy']/100.)
    print('DONE')
