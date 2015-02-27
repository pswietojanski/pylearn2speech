from pylearn2.datasets.speech import SpeechDataset
from pylearn2.datasets.preprocessing_speech import ReorderByBands 
import unittest
import numpy as np

class TestSpeech(unittest.TestCase):
    def setUp(self):
        self.dataset = SpeechDataset(flist='train.flist', 
                                     aligns='aligns.pdf', 
                                     template_shell_command='copy-feats-1file', 
                                     subset=100, 
                                     preprocessor=None)
        
    def test_io(self):
        "Tests that the data is loaded correctly"
        iterator = self.dataset.iterator()
        for batch in iterator:
            pass

    
def test_band_reordering(self):
        
    statics = numpy.ones((2,3))
    deltas = numpy.ones((2,3))*2
    ddeltas = numpy.ones((2,3))*3
        
    feats = numpy.concatenate([statics, deltas, ddeltas], axis=1)
        
    rbb = ReorderByBands(3, 1, tied_deltas=True)
        
    print feats
    rval = rbb.apply(feats)
    print rval
        
        
