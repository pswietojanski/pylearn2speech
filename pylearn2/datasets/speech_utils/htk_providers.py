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

import logging, random, h5py, numpy, os, gzip, cPickle, threading
import theano

from tnets.io.dataset import Recording, Dataset
from tnets.utils import PathModifier
from tnets.io.providers import ListDataProvider, make_shell_call
from StringIO import StringIO
from string import Template
#from tnets.experiments.speech.utils import normalise_matrix

class HTKFeatsProviderUtt(ListDataProvider):

    def __init__(self, files_paths_list, randomize=False, max_utt=-1, path_modifier=PathModifier(), gmean=None, gstdev=None, uttid_regex=None):
        try:
            super(HTKFeatsProviderUtt, self).__init__(files_paths_list, path_modifier, gmean=gmean, gstdev=gstdev, uttid_regex=uttid_regex)
            self.max_utt = max_utt 
            self.randomize = randomize
            if(randomize==True):
                random.shuffle(self.files_list)
        except IOError as e:
            logging.error(e)
            raise e
    
    def __iter__(self):
        return self
    
    def reset(self):
      self.index=0
      if(self.randomize==True):
          random.shuffle(self.files_list)
    
    def next(self):
        if ((self.index >= self.list_size) or (self.max_utt>0 and self.index >= self.max_utt)):
            raise StopIteration
        
        utt_path = self.files_list[self.index]
        features=None
        try:
            features = Recording.load_data_from_file(utt_path)
            features = self.normalise_matrix(features, self.gmean, self.gstdev)
        except Exception as e:
            print e
            
        self.index += 1
        
        return features, utt_path

class HTKHListFeatsProviderUtt(ListDataProvider):

    def __init__(self, files_paths_list, template_shell_call="HList -r ${SCP_ENTRY}", randomize=False, max_utt=-1, path_modifier=PathModifier(), gmean=None, gstdev=None):
        try:
            super(HTKHListFeatsProviderUtt, self).__init__(files_paths_list, path_modifier, gmean=gmean, gstdev=gstdev)
            self.max_utt = max_utt 
            self.template_shell_call = Template(template_shell_call)
            self.randomize = randomize
            self.utt_skipped=0
            if(randomize==True):
                random.shuffle(self.files_list)
        except IOError as e:
            logging.error(e)
            raise e
    
    def __iter__(self):
        return self
    
    def reset(self):
      self.index=0
      self.utt_skipped=0
      if(self.randomize==True):
          random.shuffle(self.files_list)
    
    def next(self):
        if ((self.index >= self.list_size) or (self.max_utt>0 and self.index >= self.max_utt)):
            print 'HTKHListFeatsProviderUtt: Skipped %i utterances out\
                 of total %i due to HList shell errors (missing files).'%(self.utt_skipped, len(self.files_list))
            raise StopIteration
        
        utt_path = self.files_list[self.index]
        features=None
        try:
            shell_call=self.template_shell_call.substitute(SCP_ENTRY=utt_path)
            features = Recording.load_data_from_hlist(shell_call)
        except Exception as e:
            self.utt_skipped+=1
            #print 'Shell command failed: ', e
                
        self.index += 1

        return features, utt_path

    def extract_matrix(self, input):
        labels = numpy.loadtxt(StringIO(i[1]), dtype=numpy.float32)
        return labels
        

class HTKAlignFeatsProviderUtt(ListDataProvider):
    def __init__(self, files_paths_list, dataset, randomize=False, max_utt=-1, path_modifier=PathModifier(), gmean=None, gstdev=None, uttid_regex=None):
        try:
            super(HTKAlignFeatsProviderUtt, self).__init__(files_paths_list, path_modifier, gmean=gmean, gstdev=gstdev, uttid_regex=uttid_regex)
            self.dataset = dataset
            self.max_utt = max_utt
            self.randomize = randomize
            #self.h5file = h5py.File(dataset,'r')           
            if(randomize==True):
                random.shuffle(self.files_list)
            self.skipped_aligns = 0
        except IOError as e:
            raise e

    def __iter__(self):
        return self
    
    def reset(self):
      self.index=0
      self.skipped_aligns=0
      if(self.randomize==True):
          random.shuffle(self.files_list)
    
    def next(self):
        if ((self.index >= self.list_size) or (self.max_utt>0 and self.index >= self.max_utt)):
            print 'In total, %i aligns were skipped out of %i'%(self.skipped_aligns, len(self.files_list))
            raise StopIteration
        
        utt_path = self.files_list[self.index]
        features, labels = None, None
        try:
            features = Recording.load_data_from_file(utt_path)
            try:
                h5file = h5py.File(self.dataset,'r')
                labels = numpy.asanyarray(h5file[self.extract_uttid_from_path(utt_path, self.uttid_regex)], dtype=numpy.float32)
                h5file.close()
            except Exception as e:
                self.skipped_aligns += 1
                print 'Lack of alignments for %s mapped with key %s.'\
                        %(utt_path, self.extract_uttid_from_path(utt_path, self.uttid_regex)), e
                pass
            
        except Exception as e:
            print e
            
        self.index += 1
        
        if (features!=None and labels!=None):
            feats_frames, labs_frames = features.shape[0], labels.shape[0]
            if feats_frames != labs_frames: 
                if (feats_frames - labs_frames)==1: #1 frame difference may happend when labels are imported from mlf, fix it here
                    labels = numpy.resize(labels, (feats_frames,)) #add an extra frame at the end
                    labels[-1] = labels[-2] #and copy the label
                else:
                    print 'Different frame numbers between feats (%i) and labels (%i) for utt %s'%\
                            (feats_frames, labs_frames, utt_path)
                    return (features, None), utt_path 
                       
        features = self.normalise_matrix(features, self.gmean, self.gstdev)
        
        return (features, labels), utt_path

    
"""