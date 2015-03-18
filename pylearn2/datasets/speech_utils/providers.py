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

import os
import subprocess
import logging
import re
import numpy

from subprocess import Popen, PIPE, CalledProcessError

log = logging.getLogger(__name__)

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
    def __init__(self, files_paths_list):
        try:
            f = open(files_paths_list, 'r')
            lines = f.readlines()
            f.close()

            self.files_list = []
            for line in lines:
                if len(line.strip()) < 1:
                    continue
                self.files_list.append(line.strip())

            self.index = 0
            self.list_size = len(self.files_list)
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

    def reset(self):
        self.index = 0

    def get_data_specs(self):
        raise NotImplementedError(str(type(self)) + " does not implement get_data_sepcs.")


class BufferedProvider(object):
    def __init__(self, provider, batch_size):
        self.provider = provider
        self.batch_size = batch_size
        self.feats_buf = []
        self.labs_buf = []
        self.max_buf_elems = 20
        self.feats_tail = None
        self.labs_tail = None
        self.finished = False

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
            if len(self.feats_buf) < 1:

                if self.finished: raise StopIteration

                tmp_feats_list, tmp_labs_list = [], []
                num_frames_read = 0

                if self.feats_tail is not None and self.labs_tail is not None:
                    tmp_feats_list.append(self.feats_tail)
                    tmp_labs_list.append(self.labs_tail)
                    num_frames_read = self.feats_tail.shape[0]
                    self.feats_tail, self.labs_tail = None, None

                try:
                    while num_frames_read < self.max_buf_elems * self.batch_size:
                        result, utt = self.provider.next()
                        if (result[0] is None) or (result[1] is None):
                            # print 'BufferedProvider: skipping %s'%utt
                            continue
                        num_frames_read += result[0].shape[0]
                        tmp_feats_list.append(result[0])
                        tmp_labs_list.append(result[1])
                except StopIteration:
                    self.finished = True

                if self.finished and num_frames_read < self.batch_size:
                    raise StopIteration

                feats = numpy.concatenate(tmp_feats_list)
                labs = numpy.concatenate(tmp_labs_list)

                assert feats.shape[0] == labs.shape[0]

                indexes = numpy.arange(self.batch_size, feats.shape[0], self.batch_size)
                self.feats_buf = numpy.split(feats, indexes)
                self.labs_buf = numpy.split(labs, indexes)

                if self.feats_buf[-1].shape[0] != self.batch_size:
                    self.feats_tail = self.feats_buf.pop()
                    self.labs_tail = self.labs_buf.pop()

            feats = self.feats_buf.pop()
            labs = self.labs_buf.pop()
            return (feats, labs)

        except StopIteration:
            raise StopIteration


class BufferedProviderDataSpec(object):
    def __init__(self, provider, batch_size):
        self.provider = provider
        self.batch_size = batch_size
        self.max_buf_elems = 5
        self.finished = False
        self.buffer = []
        self.buffer_tail = None
        self.provider.reset()

        self.space, self.sources = self.provider.get_data_specs()

    def reset(self):
        self.provider.reset()
        self.finished = False

    def __iter__(self):
        return self

    def next(self):
        try:

            if len(self.buffer) < 1:

                if self.finished:
                    raise StopIteration

                num_frames_read = 0
                tmp_buffer = []
                if self.buffer_tail is not None:
                    tmp_buffer.append(self.buffer_tail)
                    num_frames_read += self.buffer_tail[0].shape[0]
                    self.buffer_tail = None

                try:
                    while num_frames_read < self.max_buf_elems * self.batch_size:
                        utt = self.provider.next()
                        if any(u is None for u in utt) or len(utt) != len(self.sources):
                            #log.warning('Data specs expected to be %s but provider'
                            #            'returned %i spaces and at least one of them'
                            #            'is None. This could be a broken example'
                            #            '(missing targets or features). Skipping.'
                            #            % (self.sources, len(utt)))
                            continue
                        # TODO: add assert checking if shape[0] of all u is the same
                        num_frames_read += utt[0].shape[0]
                        tmp_buffer.append(utt)
                except StopIteration:
                    self.finished = True

                if self.finished and (0 < num_frames_read < self.batch_size):
                    log.warning('Skipped last %i random elements in dataset '
                                'since it did not fit into full batch of '
                                'size %i (they are likely to be presented '
                                'in the next iteration)' %
                                (num_frames_read, self.batch_size))
                    raise StopIteration

                # transpose [ (x1,y1,...), (x2,y2,...), ...]
                # to [ (x1, x2, ...), (y1, y2, ...), ...]
                # so we can concatenate corresponding spaces and
                # create mini-batches of specified size 'batch_size'
                trans = zip(*tmp_buffer)
                assert len(trans) == len(self.sources), (
                    "Unzipped list of unexpected length %i instead of %i spaces." % \
                    (len(trans), len(self.sources))
                )

                batched_data = []
                for space in trans:
                    space_data = numpy.concatenate(space, axis=0)
                    indexes = numpy.arange(self.batch_size, space_data.shape[0], self.batch_size)
                    space_data_mbatches = numpy.split(space_data, indexes)
                    # TODO: any other way using in-built function can do this in a more efficient/elegant way?
                    #from a list of numpy arrays need to make a list of tuples of numpy arrays
                    #so each element on the list is a tuple that contains a batch_size of
                    # elements in format accepted by data_specs, most likely something like
                    # [(ndarray, ndarray, ...), (ndarray, ndarray, ...), ... ]
                    batched_data.append(space_data_mbatches)

                self.buffer = zip(*batched_data)

                # self.buffer is now [(x,y,z),(x,y,z), where x,y,z are nd array each
                # at least x is always expected to exists, so it is safe to address [0]
                if self.buffer[-1][0].shape[0] != self.batch_size:
                    self.buffer_tail = self.buffer.pop()

            rval = self.buffer.pop()
            self._validate_batch(rval)

            return rval

        except StopIteration:
            raise StopIteration

    def _validate_batch(self, batch):
        if not isinstance(batch, tuple):
            raise TypeError("The value of batch is expected to be tuple"
                            "but got %s of type" % (batch, type(batch)))
        if len(batch) != len(self.sources):
            raise ValueError("Expected %d elements in batch, got %d"
                             % (len(self.sources), len(batch)))

    def num_classes(self):
        return self.provider.num_classes()
