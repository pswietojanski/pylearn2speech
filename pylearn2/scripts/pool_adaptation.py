#!/usr/bin/env python

__author__ = "Pawel Swietojanski"

import os, sys, numpy, struct, tempfile, theano, time, re

from optparse import OptionParser
from theano import function
from theano import tensor as T

from pylearn2.datasets.speech_utils.kaldi_providers import read_ark_entry_from_buffer, write_ark_entry_to_buffer
from pylearn2.scripts.pkl_to_pytables import ModelPyTables
from pylearn2.utils import serial, sharedX

def fill_adapt_template(yaml_tpl, dict):
    yaml_template = open(yaml_tpl, 'r').read()
    

def main(args=None):
    
    usage = "pool_adaptation.py [options] <si-model-dir> <sa-model-dir> <feats-scp> <targets-pdf> "
    
    parser = OptionParser()
    parser.add_option("--adapt-yaml", dest="adapt_yaml", default="",
                      help="Provide the adaptation yaml template to start with")
    parser.add_option("--freeze-means", dest="freeze_means", default=False,
                      help="Skip means while updating pools")
    parser.add_option("--freeze-betas", dest="freeze_betas", default=False,
                      help="Skip precisions while updating pools")
    parser.add_option("--freeze-amp", dest="freeze_amp", default="true",
                      help="Skip activation function amplitudes while updating pools")
    parser.add_option("--freeze-slopes", dest="freeze_slopes", default="true",
                      help="Skip activation function slopes while updating pools")
    parser.add_option("--freeze-layer-ids", dest="freeze_layer_ids", default="",
                      help="update params only in this layers, i.e. --layer-ids 012")
    parser.add_option("--job", dest="JOB", default=0,
                      help="JOB ID used to store model in")
    parser.add_option("--debug", dest="debug", default=False,
                      help="Prints activations and shapes in text format rather than binary Kaldi archives")
    
    (options,args) = parser.parse_args(args=args)
    
    print options.adapt_yaml
    print options.freeze_means
    print options.freeze_betas
    print options.freeze_amp
    print options.freeze_slopes
    print options.freeze_layer_ids
    print options.JOB
    print 'ARGS: ', args
   
    #if options.adapt_yaml!='':
    #    NotImplementedError('Lodaing from pkl not yet supported due to GPU/CPU pickle issues.')
         
    if len(args) != 5:
        print usage
        exit(1)
        
    si_model_dir = args[1]
    sa_model_dir = args[2]
    feats_scp = args[3]
    targets_pdf = args[4]

    #print "si model dir is %s"%si_model_dir   
    model_yaml = "%s/adapt_final%s.yaml"%(si_model_dir, options.JOB)
    model_params = "%s/cnn_best.h5"%si_model_dir
    
    #print 'Yaml path', model_yaml
    #print 'Params path', model_params
    
    if not os.path.isfile(options.adapt_yaml):
        raise Exception('File %s not found'%options.adapt_yaml)
    if not os.path.isfile(model_params):
        raise Exception('File %s not found'%model_params)
    
    vars={}
    vars['adapt_flist']=feats_scp
    vars['adapt_pdfs']=targets_pdf
    vars['adapt_lr']=0.05
    vars['adapt_momentum']=0.5
    vars['sa_dir'] = sa_model_dir
    vars['JOB'] = options.JOB

    #print vars
    #print 'Locals: ',locals()   

    adapt_template = open(options.adapt_yaml, 'r').read()
    adapt_template_str = adapt_template % vars
    f = open(model_yaml, 'w')
    f.write(adapt_template_str)
    f.close()

    print 'Building model %s'%model_yaml
    train_obj = serial.load_train_file(model_yaml)
    print 'Loading params from %s'%model_params
    params = ModelPyTables.pytables_to_params(model_params, name='Model')
    train_obj.model.set_params(params)

    freeze_regex='softmax_[Wb]|h[0-9]_[Wb]|nlrf_[Wb]'    
    
    if options.freeze_layer_ids!='':
       layers = options.freeze_layer_ids
       freeze_regex = "%s|g[%s]p_u|g[%s]p_beta"%(freeze_regex, layers, layers)

    if options.freeze_means == 'true':
        freeze_regex = "%s|g[0-9]p_u"%(freeze_regex)
    if options.freeze_betas == 'true':
        freeze_regex = "%s|g[0-9]p_beta"%(freeze_regex)
    if options.freeze_amp == 'true':
        freeze_regex = "%s|g[0-9]p_amp"%(freeze_regex)
    if options.freeze_slopes == 'true':
        freeze_regex = "%s|g[0-9]p_arg"%(freeze_regex)

    #print "Freeze regex is", freeze_regex

    model_params = train_obj.model.get_params()    

    params_to_freeze = {}
    for param in model_params:
        if re.match(freeze_regex, str(param)) is not None:
             if param not in params_to_freeze:
                 params_to_freeze[param] = param

    #print params_to_freeze
    if len(params_to_freeze)==len(model_params):
        print 'None of the parameters were set to be updated. Freeze list is', params_to_freeze
        exit(0)
 
    train_obj.model.freeze(params_to_freeze.values())
    print 'Will update those params only: ', train_obj.model.get_params()
    train_obj.main_loop()
    train_obj.model.freeze_set = set([]) #unfreeze so get_params will return all model params
    print 'Unfreezed params are ', train_obj.model.get_params()
    
    
if __name__=="__main__":
    main(sys.argv)
