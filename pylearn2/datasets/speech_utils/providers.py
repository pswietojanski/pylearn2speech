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

import os, subprocess, logging, re, numpy

from subprocess import Popen, PIPE, CalledProcessError
from pylearn2.utils.speech_tmp import PathModifier

def make_shell_call(shell_call):
    buffer_tuple = (None, None)
    try:
        buffer_tuple = Popen(args=shell_call, bufsize=-1, \
                    stdout=PIPE, stderr=PIPE, shell=True).communicate()
        return buffer_tuple
    except CalledProcessError as cpe:
        print 'CPE', cpe
        return None
    except OSError as oe:
        print 'OE', oe
        return None
    except ValueError as ve:
        print 'VE', ve
        return None
    return None

class ListDataProvider(object):

    def __init__(self, files_paths_list, path_modifier=PathModifier(), gmean=None, gstdev=None, uttid_regex=None):
        self.path_modifier = path_modifier
        try:
            f = open(files_paths_list, 'r')
            lines = f.readlines()
            f.close()
            
            #remove potential empty lines and end of line signs
            self.files_list = []
            for line in lines:
                if len(line.strip()) < 1:
                    continue
                self.files_list.append(self.path_modifier.get_path(line.strip()))
            
            self.index = 0
            self.list_size = len(self.files_list)
            self.uttid_regex = uttid_regex
            self.gmean = gmean
            self.gstdev = gstdev
        except IOError as e:
            logging.error(e)
            raise e
    
    def __iter__(self):
        return self 
    
    def next(self):
        if self.index >= self.list_size:
            raise StopIteration
        utt_path = self.files_list[self.index]           
        self.index += 1
        return utt_path
    
    def normalise_matrix(self, features, gmean=None, gstdev=None, mean=0.0, var=1.0):
        if (gmean is None and gstdev is None):
            return features
        elif (gmean != None and gstdev is None):
            return (features - gmean) + mean
        else: 
            return (features - gmean)/gstdev*var + mean
       
    def generate_flist(self, output_fname):
        fout = open(output_fname, 'w')
        for path in self.files_list:
            print >> fout, path 
        fout.close()
    
    def generate_map_flist(self, output_fname, rel2abs=False):
        fout = open(output_fname, 'w')
        for path in self.files_list:
            if rel2abs: path = os.path.abspath(path)
            print >> fout, self.extract_uttid_from_path(path)+' '+path 
        fout.close()
        
    def reset(self):
        self.index = 0
    
    @staticmethod
    def extract_uttid_from_path(path, uttid_regex=None):
        #TODO Move this regex to some sort of config and compile it once!
        if uttid_regex is None:
            #by default assume filename without extension
            uttid_regex = '(.*)\..*'      
        uttid_re = re.compile(r'%s'%uttid_regex)
        fname =  os.path.basename(path)
        mm = uttid_re.match(fname)
        if mm!=None:           
            return mm.group(1)
        else:
            print 'Cannot match UTTID using %s from %s !'%(uttid_regex, fname)
            return None
        #return (os.path.basename(path).split('.')[0])[4:]
        

class BufferedProvider(object):
    def __init__(self, provider, batch_size):
        self.provider = provider
        self.batch_size = batch_size
        self.feats_buf = []
        self.labs_buf = []
        self.max_buf_elems = 20 
        self.feats_tail=None
        self.labs_tail=None
        self.finished=False
        
        self.provider.reset()
        
    def reset(self):
        self.provider.reset()
        self.finished = False
    
    @property
    def num_classes(self):
        return self.provider.num_classes
      
    def __iter__(self):
        return self
    
    def next(self):
        try:
            if (len(self.feats_buf)<1):
                
                if self.finished: raise StopIteration
                
                tmp_feats_list, tmp_labs_list = [], []
                num_frames_read=0
                
                if self.feats_tail != None and self.labs_tail != None:
                    tmp_feats_list.append(self.feats_tail)
                    tmp_labs_list.append(self.labs_tail)
                    num_frames_read = self.feats_tail.shape[0]
                    self.feats_tail, self.labs_tail = None, None
                
                try:
                    while num_frames_read < self.max_buf_elems*self.batch_size:
                        result, utt = self.provider.next()
                        if (result[0] is None) or (result[1] is None):
                            #print 'BufferedProvider: skipping %s'%utt
                            continue
                        num_frames_read += result[0].shape[0]
                        tmp_feats_list.append(result[0])
                        tmp_labs_list.append(result[1])
                except StopIteration:
                    self.finished = True
                
                if self.finished and num_frames_read<self.batch_size:
                    raise StopIteration
                
                feats = numpy.concatenate(tmp_feats_list)
                labs = numpy.concatenate(tmp_labs_list)
                
                assert feats.shape[0] == labs.shape[0]
                
                indexes = numpy.arange(self.batch_size, feats.shape[0], self.batch_size)
                self.feats_buf=numpy.split(feats, indexes)
                self.labs_buf=numpy.split(labs, indexes)

                if self.feats_buf[-1].shape[0]!=self.batch_size:
                    self.feats_tail = self.feats_buf.pop()
                    self.labs_tail = self.labs_buf.pop()

            feats = self.feats_buf.pop()
            labs = self.labs_buf.pop()
            return (feats, labs)
               
        except StopIteration:
            raise StopIteration

