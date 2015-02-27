#!/usr/bin/env python
__author__ = "Pawel Swietojanski"

import sys, types, tables, numpy, theano
from pylearn2.utils import serial
from theano import shared

class ModelPyTables(object):
    
    filters = tables.Filters(complib='zlib', complevel=1)
    
    def __init__(self):
        pass

    @staticmethod
    def model_to_pytables(path, model, name='Model', desc=''):
        """
        Initialize hdf5 file to store the model
        """
               
        h5file = tables.openFile(path, mode = "w", title = "Model parameters file")
        gcolumns = h5file.createGroup(h5file.root, name, desc)
        filters = ModelPyTables.filters
        
        model.freeze_set = set([])
        params = model.get_params()
        for param in params:
            p_value = param.get_value()
            p_atom = tables.Atom.from_dtype(p_value.dtype)
            p_array = h5file.createCArray(gcolumns, param.name, atom = p_atom, shape = p_value.shape,
                                title = param.name, filters = filters)
            p_array[:] = p_value
            h5file.flush()
            print 'ModelPyTables: exporting param %s with shape %s and dtype %s'%(param.name, p_value.shape, p_value.dtype)
        
        h5file.close()     
       
    @staticmethod
    def pytables_to_params(path, name='Model'):
        """Returns dictionary {'param_name': value} so the model can appropriately set these parameters back.
        Bear in mind this is just a dictionary of ndarrays. These should be then loaded into an appropriate Theano variables.
        The advantage is those (theano variables) could be easily build for desired backend (GPU, CPU) first so Ian hacky pickle
        conversion is not required in this case."""
        
        params = {}
        h5file = tables.openFile(path, mode = "r")
        for node in h5file.walkNodes('/%s'%name, "Array"):
            if params.has_key(node.name): 
                raise KeyError('Key already exists %s'%node.name) #it should not happen, but check anyway
            params[node.name] = node.read()
            #print 'ModelPyTables: Lodaing param %s into dictionary (shape is %s and dtype %s)'%\
            #                                    (node.name, params[node.name].shape, params[node.name].dtype)
        h5file.close()
        return params

if __name__=="__main__":

    assert len(sys.argv)==3
    #hardcoded_layer_names = {"softmax":"y"}
    _,pkl_path_to, pytable_path = sys.argv
    
    model = serial.load(pkl_path_to)
    ModelPyTables.model_to_pytables(pytable_path, model, name="Model", desc="Model parameters exported from pickle %s"%pkl_path_to)
