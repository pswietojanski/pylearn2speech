import os, sys, numpy, subprocess, struct, tempfile, theano, time

from optparse import OptionParser
from theano import function
from theano import tensor as T

from pylearn2.datasets.speech_utils.kaldi_providers import read_ark_entry_from_buffer, write_ark_entry_to_buffer
from pylearn2.scripts.pkl_to_pytables import ModelPyTables
from pylearn2.utils import serial, sharedX
from pylearn2.datasets.preprocessing_speech import ReorderByBands

class Pylearn2KaldiDecoderProvider(object):
    def __init__(self, model, preprocessor=None):
        self.model = model
        self.preprocessor = preprocessor
        self.fprop = None
    
    def apply_preprocessor(self, x):
        rval = x
        if self.preprocessor!=None:
            rval = self.preprocessor.apply(x)
        return rval
    
    def set_model_params(self, params):
        self.model.set_params(params)

def instantiate_model(yaml_path):
    from pylearn2.models.model import Model
    model = serial.load_train_file(yaml_path)
    assert isinstance(model, Model)
    return model

def instantiate_decoder_provider(yaml_path):
    
    from pylearn2.models.model import Model
    from pylearn2.datasets.preprocessing_speech import OnlinePreprocessor
    
    decoder = serial.load_train_file(yaml_path)
    
    assert isinstance(decoder, Pylearn2KaldiDecoderProvider)
    assert isinstance(decoder.model, Model)
    assert isinstance(decoder.preprocessor, OnlinePreprocessor) or decoder.preprocessor is None
    
    return decoder

def load_model_pytable(yaml_path, pytable_path, name='Model'):
    model = instantiate_model(yaml_path)
    params = ModelPyTables.pytables_to_params(pytable_path, name=name)
    model.set_params(params)
    return model

def load_decoder_provider(yaml_path='', pytable_path='', pkl_path='', name='Model'):

    if pkl_path!='':
        model = serial.load(pkl_path)
        #preprocessor = ReorderByBands(23, 11) #fix this!!!
        preprocessor = None
        decoder = Pylearn2KaldiDecoderProvider(model, preprocessor=preprocessor)
    else:
        decoder = instantiate_decoder_provider(yaml_path)
        params = ModelPyTables.pytables_to_params(pytable_path, name=name)
        decoder.set_model_params(params)

    return decoder

def load_kaldi_priors(path):
    numbers=numpy.fromregex(path, r"(\d+)", dtype=[('num', numpy.int32)])
    class_counts=numpy.asarray(numbers['num'], dtype=theano.config.floatX)
    priors = class_counts/class_counts.sum()
    #floor zeroes to something small so log() on that will be different from -inf or better skip these in contribution at all i.e. set to -log(0)?
    priors[priors<1e-10] = 1e-10 
    assert numpy.all(priors > 0) and numpy.all(priors <= 1.0)
    log_priors = numpy.log(priors)
    assert not numpy.any(numpy.isinf(log_priors))
    return log_priors

def init_environment(yaml_path, pytable_path, pkl_path='', priors_path=''):

    decoder = load_decoder_provider(yaml_path, pytable_path, pkl_path)
    
    priors, log_priors = None, None
    if priors_path!='':
        priors = load_kaldi_priors(priors_path)
        log_priors = sharedX(priors, name='log_priors')
    
    z = T.matrix('outputs')
    lp = T.vector('lp')
        
    X = decoder.model.get_input_space().make_batch_theano()
    X.name = 'X'
    y = decoder.model.fprop(X)
    y.name = 'y'
        
    z = T.log(y)
    
    if log_priors != None:
        z = z - lp
        fprop = function(inputs = [X], 
                         outputs = z, 
                         givens = {lp: log_priors})
    else:
        fprop = function([X], z)
    
    decoder.fprop = fprop

    return decoder

def main_loop(buffer, decoder, debug=False):
    
    #Changing batch_size on the fly do not work with ConvOp for some reason
    #this function splits the data into original batch sizes and padds the last minbatch 
    def make_batches(feats, batch_size):
        indexes = numpy.arange(batch_size, feats.shape[0], batch_size)
        flist = numpy.split(feats, indexes)
        lb = flist[-1]
        if lb.shape[0] != batch_size:
            new_shape = tuple([batch_size] + list(lb.shape[1:]))
            tmp = numpy.zeros(new_shape, dtype=lb.dtype)
            tmp[0:lb.shape[0]] = lb
            flist[-1] = tmp
        return flist
    
    while True:
        
        rval  = read_ark_entry_from_buffer(buffer)
        if rval=='':
           break
        uttid, feats = rval
        
        pfeats = decoder.apply_preprocessor(feats)  #preprocess if necessary
        #decoder.model.set_batch_size(pfeats.shape[0])
        batches = make_batches(pfeats, decoder.model.batch_size)
        
        factivations = []
        for batch in batches:
            factivations.append(decoder.fprop(batch))
        
        activations = numpy.concatenate(factivations, axis=0)
        activations = activations[0:pfeats.shape[0],:] #drop the padded part
        
        assert activations.shape[0] == pfeats.shape[0]
        
        if debug:
            print feats.shape
            print pfeats.shape
            print activations.shape
            print activations
        else:
            f=tempfile.SpooledTemporaryFile(max_size=209715200) #keep up to 200MB in memory
            write_ark_entry_to_buffer(f, uttid, activations)
            f.flush()
            f.seek(0)
            print f.read()
            f.close()

def main(args=None):

    parser = OptionParser()

    parser.add_option("--model-pkl", dest="model_pkl", default="",
                      help="specify the model pickle to be used in forward pass")
    parser.add_option("--model-yaml", dest="model_yaml", default="",
                      help="specify the model yaml structure to be used in forward pass")
    parser.add_option("--model-pytables", dest="model_pytables", default="",
                      help="specify the pytable file from which weights should be loaded")
    parser.add_option("-t", "--feats-flist", dest="feats_flist", default="",
                      help="specify a feats scp to fwd pass (if not specified, pipe assumed)")
    parser.add_option("-p", "--priors", dest="priors_path", default="",
                      help="specify the priors to obtain scaled-likelihoods")
    parser.add_option("--debug", dest="debug", default=False,
                      help="Prints activations and shapes in text format rather than binary Kaldi archives")
    
    (options,args) = parser.parse_args(args=args)

    buffer = sys.stdin
    
    if (options.model_pkl!=''):
        decoder = init_environment(options.model_yaml, options.model_pytables, options.model_pkl, options.priors_path)
    elif (options.model_pytables!='' and options.model_yaml!=''):    
        decoder = init_environment(options.model_yaml, options.model_pytables, options.model_pkl, options.priors_path)
    else:
         NotImplementedError('Cannot load the model')
    
    debug = False
    if options.debug=='True':
        debug = True
    
    #TODO: load it from yaml file
    main_loop(buffer, decoder, debug=debug)

if __name__=='__main__':
    main()

