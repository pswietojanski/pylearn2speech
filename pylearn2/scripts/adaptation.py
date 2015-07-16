#!/usr/bin/env python

__author__ = "Pawel Swietojanski"

import os
import sys
import re
import logging

from argparse import ArgumentParser
from pylearn2.utils import serial, sharedX

log = logging.getLogger(__name__)

def load_params_from_list_of_hdfs(files_list, override=True):
    assert files_list is not None and files_list != "", (
        "Expected to at least one path hdf with parameters"
    )

    if isinstance(files_list, (list, tuple)):
        files = files_list
    else:
        files = re.split('[\s,]', files_list)

    params_list = []
    for f in files:
        if not os.path.isfile(f):
            raise IOError('File %s not found'%f)
        params_file = serial.load_params_from_pytables(f, container_name=None)
        params_list.append(params_file)

    rval = {}
    for idx, params in enumerate(params_list):
        for key in params.keys():
            if key in rval:
                if override:
                    log.warning('Overriding param(%s) from (%s) by the one found in (%s)'\
                                % (key, files[idx-1], files[idx]))
                    print 'Overriding param(%s) from (%s) by the one found in (%s)'\
                                % (key, files[idx-1], files[idx])
                else:
                    raise KeyError("Param %s found in both %s and %s and override is False"\
                                     % (key, files[idx-1], files[idx]))
            rval[key] = params[key]

    return rval

def main(args=None):
    
    usage = "adaptation.py [options] <si-model-dir> <sa-model-dir> <feats-scp> <targets-pdf> "
    
    parser = ArgumentParser(usage=usage)
    parser.add_argument("--adapt-yaml", dest="adapt_yaml", default="",
                      help="Provide the adaptation yaml template to start with")
    parser.add_argument("--model-pytables", dest="model_pytables", nargs='+', default=None,
                      help="A list of hdf containers with parameters, will be set in order as they appear on the list.")
    parser.add_argument("--freeze-regex", dest="freeze_regex", default="softmax_[Wb]|h[0-9]_[Wb]|nlrf_[Wb]",
                      help="Regex to use when matching parameters to freeze")
    parser.add_argument("--job", dest="JOB", default=0,
                      help="JOB ID used to store model in")
    parser.add_argument("--override", dest="override", default=True,
                      help="When specified many files and some parameters have the same name, the ones appearing later in the list will override the former ones.")
    parser.add_argument("--debug", dest="debug", default=False,
                      help="Prints activations and shapes in text format rather than binary Kaldi archives")

    parser.add_argument("si_model_dir", help="Model directory")
    parser.add_argument("sa_model_dir", help="Adaptation directory")
    parser.add_argument("feats_scp", help="Scp Kaldi list of files used to adaptation")
    parser.add_argument("targets_pdf", help="Kaldi 0-indexed list of pdfs")

    options = parser.parse_args()

    model_yaml = "%s/adapt_final%s.yaml"%(options.sa_model_dir, options.JOB)
    if not os.path.isfile(options.adapt_yaml):
        raise Exception('File %s not found'%options.adapt_yaml)
    
    vars={}
    vars['adapt_flist']=options.feats_scp
    vars['adapt_pdfs']=options.targets_pdf
    vars['adapt_lr']=0.05
    vars['adapt_momentum']=0.5
    vars['sa_dir'] = options.sa_model_dir
    vars['JOB'] = options.JOB

    print options.model_pytables

    adapt_template = open(options.adapt_yaml, 'r').read()
    adapt_template_str = adapt_template % vars
    f = open(model_yaml, 'w')
    f.write(adapt_template_str)
    f.close()

    log.info('Building model %s'%model_yaml)
    train_obj = serial.load_train_file(model_yaml)
    log.info('Loading params from %s'%options.model_pytables)
    params = load_params_from_list_of_hdfs(options.model_pytables, options.override)
    train_obj.model.set_params(params)

    model_params = train_obj.model.get_params()    

    params_to_freeze = {}
    for param in model_params:
        if re.match(options.freeze_regex, str(param)) is not None:
             if param not in params_to_freeze:
                 params_to_freeze[param] = param

    #print params_to_freeze
    if len(params_to_freeze) == len(model_params):
        log.warning('None of the parameters were set to be updated. Freeze list is', params_to_freeze)
        exit(0)
 
    train_obj.model.freeze(params_to_freeze.values())
    log.info('Will update those params only: ', train_obj.model.get_params())
    train_obj.main_loop()
    train_obj.model.freeze_set = set([]) #unfreeze so get_params will return all model params
    log.info('Unfreezed params are ', train_obj.model.get_params())
    
    
if __name__=="__main__":
    main(sys.argv)
