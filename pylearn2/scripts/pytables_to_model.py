#!/usr/bin/env python
__author__ = "Pawel Swietojanski"

import sys, types, tables, numpy, theano
from pylearn2.scripts.pkl_to_pytables import ModelPyTables
from pylearn2.utils import serial
from theano import shared

def create_model(yaml_path):
    from pylearn2.models.model import Model
    model = serial.load_train_file(yaml_path)
    assert isinstance(model, Model)
    return model
  
if __name__=="__main__":

    assert len(sys.argv)==3
    #hardcoded_layer_names = {"softmax":"y"}
    _, yaml_path, pytables_path_to = sys.argv
    
    model = create_model(yaml_path)
    params = ModelPyTables.pytables_to_params(pytables_path_to, name="Model")
    model.set_params(params)
