'''
Copyright 2011-2013 Pawel Swietojanski

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
MERCHANTABLITY OR NON-INFRINGEMENT.
See the Apache 2 License for the specific language governing permissions and
limitations under the License.
'''

"""
import numpy, os
from pylearn2.datasets.speech_utils.htk_providers import HTKFeatsProviderUtt, ListDataProvider, HTKAlignFeatsProviderUtt
from pylearn2.datasets.speech_utils.htkio import HTK_Parm_IO
from pylearn2.utils.speech_tmp import PathModifier

import theano
import theano.tensor as T

def get_global_statistics(files_list, ddof=0, var_floor=0.001):
    # get a single element to initialise size of containers
    for features, utt_path in HTKFeatsProviderUtt(files_list, max_utt=1):
        if features is None:
            print utt_path
            continue
        vec_size = features.shape[1]
        
    mean_acc = numpy.zeros((vec_size))
    sqr_acc = numpy.zeros((vec_size))
    norm = 0.0
    
    #calculate global mean and st. deviation accumulators
    #test = []
    for features, utt_path in HTKFeatsProviderUtt(files_list):
        norm += features.shape[0]
        mean_acc += numpy.sum(features, axis=0)
        sqr_acc += numpy.sum(features**2, axis=0)
        #print utt_path
        #test.append(features)
        
    gmean = mean_acc/(norm-ddof)
    gstdev = numpy.sqrt(sqr_acc/(norm-ddof) - gmean**2)
    
    #floor variance according to var_floor
    gstdev[numpy.nonzero(gstdev < var_floor)[0]] = var_floor
    
    #mtr = numpy.concatenate(test, axis=0)
    #print 'Global mean ', gmean
    #print 'Mean ', numpy.mean(mtr, axis=0)
    #print 'Global st. deviation ', gstdev
    #print 'SD ', numpy.std(mtr, axis=0)
    
    return gmean, gstdev

def get_global_statistics_dp(data_provider, ddof=0, var_floor=0.001):
    ''' Function calculates and returns global mean and standard deviation of HTK
    files listed in files_list file.
    '''
    # get a single element to initialise size of containers
    for features, utt_path in data_provider:
        vec_size = features.shape[1]
        break
        
    mean_acc = numpy.zeros((vec_size))
    sqr_acc = numpy.zeros((vec_size))
    norm = 0.0
    
    #calculate global mean and st. deviation accumulators
    #test = []
    data_provider.reset()
    for features, utt_path in data_provider:
        if features is None:
            continue
        norm += features.shape[0]
        mean_acc += numpy.sum(features, axis=0)
        sqr_acc += numpy.sum(features**2, axis=0)
        #print utt_path
        #test.append(features)
        
    gmean = mean_acc/(norm-ddof)
    gstdev = numpy.sqrt(sqr_acc/(norm-ddof) - gmean**2)
    
    #floor variance according to var_floor
    gstdev[numpy.nonzero(gstdev < var_floor)[0]] = var_floor
    
    #mtr = numpy.concatenate(test, axis=0)
    #print 'Global mean ', gmean
    #print 'Mean ', numpy.mean(mtr, axis=0)
    #print 'Global st. deviation ', gstdev
    #print 'SD ', numpy.std(mtr, axis=0)
    
    return gmean, gstdev

def normalise_matrix(features, gmean=None, gstdev=None, mean=0.0, var=1.0):
    if (gmean is None and gstdev is None):
        return features
    elif (gmean != None and gstdev is None):
        return (features - gmean) + mean
    else: 
        return (features - gmean)/gstdev*var + mean

def normalise_list(features, gmean=None, gstdev=None):
    num_frames = 0
    if (gmean is None or gstdev is None):
        for i in xrange(len(features)):
            num_frames += features[i].shape[0]
        return features, num_frames
    
    for i in xrange(len(features)):
        features[i] = normalise_matrix(features[i], gmean, gstdev)
        num_frames += features[i].shape[0]
    
    return features, num_frames

def normalise_files(files_list, mean=0.0, var=1.0, ddof=0):
    '''Function normalise list of HTK param files to have a given mean and variance
    (by default zero mean and unit variance). Parameter DDoF (delta degree of freedom)
    allows to obtain an unbiased standard deviation estimator.
    '''
    
    gmean, gstdev = get_global_statistics(files_list, ddof)
    for features, utt_path in HTKFeatsProviderUtt(files_list):
        #print utt_path
        features = normalise_matrix(features, gmean, gstdev, mean, var)
        save_as_user_htk(utt_path.strip(), numpy.asanyarray(features, dtype=numpy.float32))
    return gmean, gstdev

def count_frames(files_list):
    '''Function normalise list of HTK param files to have a given mean and variance
    (by default zero mean and unit variance). Parameter DDoF (delta degree of freedom)
    allows to obtain an unbiased standard deviation estimator.
    '''
    count = 0
    for features, utt_path in HTKFeatsProviderUtt(files_list):
        count += features.shape[0]
    return count

def generate_minibatch(features, offset, add_paddings=True, reorder_by_bands=False, num_bands=None):
    '''
    Function creates a batch to use with MLP/DNNs by adding to each row 
    an appropriate frame context defined by the offset and optional padding frames
    at the beginning and at the end of an array.
    
    :type features: numpy.ndarray
    :param features: 2D acoustic vectors num_frames x vector_size
    
    :type return: numpy.ndarray
    :param return:  the transformed features 2D array
    
    '''
    
    if offset<1 and reorder_by_bands is False:
        return features;
    
    num_frames, vec_size = features.shape
    ctx_win = numpy.arange((offset*2+1) * vec_size)
    frames = numpy.arange(num_frames) * vec_size
    indexes  = frames[:, numpy.newaxis] + ctx_win[numpy.newaxis, :]
    
    # this part reorders the dimensions in a way statics, deltas, etc. are grouped together so
    # are more convenient to use by contigous filters in convolutional network, i.e frame
    # f0(t) f1(t) ... fn(t) f0_d(t) f1_d(t) ... fn_d(t) | f0(t+1) f1(t+1) ... fn(t+1) f0_d(t+1) f1_d(t+1) ... fn_d(t+1)
    # is converted into
    # f0(t) f0(t+1) f0_d(t) f0_d(t+1) f1(t) f1(t+1) f1_d(t) f1_d(t+1) ... fn(t) fn(t+1) fn_d(t) fn_d(t+1)
    if reorder_by_bands:
        assert num_bands != None
        assert vec_size % num_bands == 0
        indexes_tmp = zeros_like(indexes)
        last_frame = indexes.shape[1]
        band_stride = vec_size / num_bands
        for i in xrange(0, num_bands):
            indexes_tmp[:,i*band_stride:(i+1)*band_stride] = indexes[:,i:last_frame:num_bands]
        indexes = indexes_tmp
        
    inputs = features.flatten()
    if (add_paddings):
        padd_beg = numpy.tile(inputs[0:vec_size], offset)
        padd_end = numpy.tile(inputs[-vec_size:], offset)
        inputs = numpy.concatenate((padd_beg, inputs, padd_end))
    
    return numpy.asarray(inputs[indexes], dtype=numpy.float32)

def generate_label_minibatch(labels, n_outs):
    
    binary_labels = numpy.zeros((labels.shape[0], n_outs), dtype=theano.config.floatX)
    binary_labels[numpy.arange(labels.shape[0]), numpy.cast['int32'](labels)] = 1
    
    return binary_labels

def shuffle_matrix_rows(features, labels):
    tmp = numpy.concatenate( (features, labels), axis=1)
    numpy.random.shuffle(tmp)
    return tmp[:,0:-1], tmp[:,-1]

def save_as_user_htk(path, data):
    
    htk = HTK_Parm_IO()
    n_samples, samp_num = data.shape
    htk.n_samples = n_samples
    htk.samp_period = 100000
    htk.samp_size = samp_num*data.itemsize
    htk.param_kind = htk.H_USER
    htk.data = data
    
    htk.write_htk(path)
    return None

def generate_mapflist_from_flist(flist, out_fname, path_modifier=PathModifier(), rel2abs=False):
    dp = ListDataProvider(flist, path_modifier)
    dp.generate_map_flist(out_fname, rel2abs)
    return None

def generate_flist_from_flist(flist, out_fname, path_modifier=PathModifier()): 
    dp = ListDataProvider(flist, path_modifier)
    dp.generate_flist(out_fname)
    return None
"""