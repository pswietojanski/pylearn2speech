
__authors__ = "Pawel Swietojanski"
__copyright__ = "Copyright 2013-2015, University of Edinburgh"

import sys
import numpy
import tempfile
import theano
import logging

from argparse import ArgumentParser
from theano import function
from theano import tensor as T

from pylearn2.models.model import Model
from pylearn2.datasets.preprocessing_speech import OnlinePreprocessor
from pylearn2.datasets.speech_utils.kaldi_providers import read_ark_entry_from_buffer, write_ark_entry_to_buffer
from pylearn2.utils import serial, sharedX

log = logging.getLogger(__name__)


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

    def build_fwdprop_function(self, log_priors_nmpy=None):

        assert self.model is not None, (
            "Model not initialised, cannot build forward-pass functions"
        )

        z = T.matrix('outputs')

        X = self.model.get_input_space().make_batch_theano()
        X.name = 'X'
        y = self.model.fprop(X)
        y.name = 'y'

        z = T.log(y)

        if log_priors_nmpy is not None:

            lp = T.vector('lp')
            log_priors = sharedX(log_priors_nmpy, name='log_priors')

            z = z - lp
            self.fprop = function(inputs = [X],
                             outputs = z,
                             givens = {lp: log_priors})
        else:
            self.fprop = function(inputs = [X], outputs = z)


def load_kaldi_priors(path, uniform_smoothing_scaler=0.05):
    assert 0 <= uniform_smoothing_scaler <= 1.0, (
        "Expected 0 <= uniform_smoothing_scaler <=1, got %f"%uniform_smoothing_scaler
    )
    numbers=numpy.fromregex(path, r"([\d\.e+]+)", dtype=[('num', numpy.float32)])
    class_counts=numpy.asarray(numbers['num'], dtype=theano.config.floatX)
    #compute the uniform smoothing count
    uniform_priors = numpy.ceil(class_counts.mean()*uniform_smoothing_scaler)
    priors = (class_counts + uniform_priors)/class_counts.sum()
    #floor zeroes to something small so log() on that will be different from -inf or better skip these in contribution at all i.e. set to -log(0)?
    flooring=1e-9
    priors[priors<flooring] = flooring
    assert numpy.all(priors > 0) and numpy.all(priors <= 1.0), (
        "Prior probabilities outside [0,1] range."
    )
    log_priors = numpy.log(priors)
    assert not numpy.any(numpy.isinf(log_priors)), (
        "Log-priors contain -inf elements."
    )
    return log_priors

def instantiate_decoder_from_yaml(yaml_filepath):

    decoder = serial.load_train_file(yaml_filepath)
    
    assert isinstance(decoder, Pylearn2KaldiDecoderProvider)
    assert isinstance(decoder.model, Model)
    assert isinstance(decoder.preprocessor, OnlinePreprocessor) or decoder.preprocessor is None
    
    return decoder

def prepare_decoder(decoder_yaml_filepath):
    if decoder_yaml_filepath is not None:
        decoder = instantiate_decoder_from_yaml(decoder_yaml_filepath)
    else: #instantiate just empty decoder object
        decoder = Pylearn2KaldiDecoderProvider(model=None, preprocessor=None)
    return decoder

def load_model_params_from_pytables(model_pytables_filepath,
                                    model_pytables_sd_filepath):

    params = {}
    #load general parameters (speaker independent)
    if model_pytables_filepath is not None:
        params = serial.load_params_from_pytables(model_pytables_filepath,
                                                  container_name="Model")
    #and possibly load also speaker dependent ones, which override the
    #speaker independent ones when the names are the same
    if model_pytables_sd_filepath is not None:
        params_sd = serial.load_params_from_pytables(model_pytables_sd_filepath)
        for sd_key in params_sd.keys():
            if sd_key in params.keys():
                log.warning("Key %s appears in both containers %s and %s."
                            "Seaker-dependent ones will override "
                            "speaker-independent in the final model"
                            %(sd_key, model_pytables_filepath,
                              model_pytables_sd_filepath))
            params[sd_key] = params_sd[sd_key]

    return params

def prepare_decoding_pipeline(options):

    #prepare decoder from yaml or simply instantiate an empty instance
    #with model and preprocessors = None
    decoder = prepare_decoder(options.decoder_yaml)
    if decoder.model is None:
        if options.model_pkl is None:
            raise Exception('Not specified how to build or load the model')
        model = serial.load(options.model_pkl)
        decoder.model = model

    if decoder.preprocessor is None:
        if options.prepr_yaml is None:
            log.warning("Preprocessor is empty, feats will be fed into the model"
                     " as provided from the pipe or as read from archive.")
            preprocessor = None
        else:
            preprocessor = serial.load_train_file(options.prepr_yaml)
        decoder.preprocessor = preprocessor

    if (options.model_pytables is not None) or \
       (options.model_pytables_sd is not None):
        params = load_model_params_from_pytables(options.model_pytables,
                                                 options.model_pytables_sd)
        #TODO: add cross-check if parameters keys match 1:1 with the model
        decoder.set_model_params(params)

    log_priors = None
    if options.priors_path is not None:
        log_priors = load_kaldi_priors(options.priors_path)

    decoder.build_fwdprop_function(log_priors)

    assert isinstance(decoder, Pylearn2KaldiDecoderProvider)
    assert isinstance(decoder.model, Model)
    assert isinstance(decoder.preprocessor, OnlinePreprocessor) or decoder.preprocessor is None
    assert decoder.fprop is not None

    return decoder

def decoder_loop(buffer, decoder, debug=False):
    
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
    #ts = numpy.zeros((11999,))
    while True:
        
        rval  = read_ark_entry_from_buffer(buffer)
        if rval=='':
           break
        uttid, feats = rval

        if not isinstance(feats, tuple):
            feats = (feats,)

        pfeats, = decoder.apply_preprocessor(feats)  #preprocess if necessary
        #decoder.model.set_batch_size(pfeats.shape[0])
        batches = make_batches(pfeats, decoder.model.batch_size)
        
        factivations = []
        for batch in batches:
            factivations.append(decoder.fprop(batch))
        
        activations = numpy.concatenate(factivations, axis=0)
        activations = activations[0:pfeats.shape[0],:] #drop the padded part
        
        assert activations.shape[0] == pfeats.shape[0]
        
        if debug:
            print "UTTID: %s\n"%uttid
            print "Original (piped) features are of shape: ", feats[0].shape
            print "Pre-processed features are of shape: ", pfeats.shape
            print "Predictions are of shape: ", activations.shape
            #tses = numpy.argmax(activations, axis=1).tolist()
            #for t in tses:
            #    ts[t] += 1
        else:
            f=tempfile.SpooledTemporaryFile(max_size=409715200) #keep up to 200MB in memory
            write_ark_entry_to_buffer(f, uttid, activations)
            f.flush()
            f.seek(0)
            print f.read()
            f.close()
    #numpy.save('ts_30h.dnn.1k', ts)

def main(args=None):

    parser = ArgumentParser()

    parser.add_argument("--decoder-yaml", dest="decoder_yaml", default=None,
                        help="specify the decoder yaml structure (including seed model) to start with")
    parser.add_argument("--model-pkl", dest="model_pkl", default=None,
                        help="specify the model pickle to be used in forward pass")
    parser.add_argument("--model-pytables", dest="model_pytables", default=None,
                        help="specify the pytable file from which weights should be loaded")
    parser.add_argument("--model-pytables-sd", dest="model_pytables_sd", default=None,
                        help="specify the pytable file with speaker dependent parameters."
                           "Note, in case the parameters overlap with --model-pytables"
                           "the si will be over-ridden by sd.")
    parser.add_argument("--prepr-yaml", dest="prepr_yaml", default=None,
                        help="Yaml describing feature pipline preprocessors.")
    parser.add_argument("--feats-flist", dest="feats_flist", default=None,
                        help="specify a feats scp to fwd pass (if not specified, Kaldi pipe assumed)")
    parser.add_argument("--priors", dest="priors_path", default=None,
                        help="specify the priors to obtain scaled-likelihoods")
    parser.add_argument("--debug", dest="debug", default=False,
                        help="Prints activations and shapes in text format rather than binary Kaldi archives")
    
    options = parser.parse_args()

    debug = False
    if options.debug=='True':
        debug = True

    decoder = prepare_decoding_pipeline(options)
    buffer = sys.stdin

    decoder_loop(buffer, decoder, debug=debug)

if __name__=='__main__':
    main()

