"""
WRITEME
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"
import os
import sys
from pylearn2.utils import serial
import logging
import warnings
from pylearn2.monitor import Monitor
from pylearn2.space import NullSpace
from pylearn2.utils.timing import log_timing
from pylearn2.utils import sharedX
import theano.tensor as T


log = logging.getLogger(__name__)


class Train(object):
    """
    A class representing the main loop of the training script.  Trains the
    specified model using the specified algorithm on the specified dataset.
    After each call to the training algorithm, the model is saved to save_path.
    May be enhanced with TrainExtension plugins.
    """
    def __init__(self, dataset, model, algorithm=None, save_path=None,
                 save_freq=0, extensions=None, allow_overwrite=True):
        """
        Construct a Train instance.

        Parameters
        ----------
        dataset : object
            Object that implements the Dataset interface defined in
            `pylearn2.datasets`.
        model : object
            Object that implements the Model interface defined in
            `pylearn2.models`.
        algorithm : object, optional
            Object that implements the TrainingAlgorithm interface
            defined in `pylearn2.training_algorithms`.
        save_path : str, optional
            Path  to save the (pickled) model.
        save_freq : int, optional
            Frequency of saves, in epochs. A frequency of zero disables
            automatic saving altogether. A frequency of 1 saves every
            epoch. A frequency of 2 saves every other epoch, etc. (default=0,
            i.e. never save)
            Note: when automatic saving is enabled (eg save_freq > 0), the
            model is always saved after learning, even when the final epoch is
            not a multiple of save_freq.
        extensions : iterable, optional
            A collection of TrainExtension objects whose callbacks are
            triggered at various points in learning.
        """
        self.allow_overwrite = allow_overwrite
        self.first_save = True
        self.dataset = dataset
        self.model = model
        self.algorithm = algorithm
        if save_path is not None:
            if save_freq == 0:
                warnings.warn('save_path specified but save_freq is 0 '
                              '(never save). Is this intentional?')
            self.save_path = save_path
        else:
            if save_freq > 0:
                phase_variable = 'PYLEARN2_TRAIN_PHASE'
                if phase_variable in os.environ:
                    phase = 'phase%d' % os.environ[phase_variable]
                    tokens = [os.environ['PYLEARN2_TRAIN_FILE_NAME'],
                              phase, 'pkl']
                else:
                    tokens = os.environ['PYLEARN2_TRAIN_FILE_NAME'], 'pkl'
                self.save_path = '.'.join(tokens)
        self.save_freq = save_freq

        if hasattr(self.dataset,'yaml_src'):
            self.model.dataset_yaml_src = self.dataset.yaml_src
        else:
            warnings.warn("dataset has no yaml src, model won't know what data it was trained on")

        self.extensions = extensions if extensions is not None else []
        self.monitor_time = sharedX(value=0,name='seconds_per_epoch')

    def setup_extensions(self):
        for ext in self.extensions:
            ext.setup(self.model, self.dataset, self.algorithm)

    def main_loop(self):
        """
        Repeatedly runs an epoch of the training algorithm, runs any
        epoch-level callbacks, and saves the model.
        """
        if self.algorithm is None:
            self.model.monitor = Monitor.get_monitor(self.model)
            self.setup_extensions()
            self.run_callbacks_and_monitoring()
            while True:
                rval = self.model.train_all(dataset=self.dataset)
                if rval is not None:
                    raise ValueError("Model.train_all should not return anything. Use Model.continue_learning to control whether learning continues.")
                self.model.monitor.report_epoch()
                if self.save_freq > 0 and self.model.monitor.epochs_seen % self.save_freq == 0:
                    self.save()
                continue_learning = self.model.continue_learning()
                assert continue_learning in [True, False, 0, 1]
                if not continue_learning:
                    break
        else:
            self.algorithm.setup(model=self.model, dataset=self.dataset)
            self.setup_extensions()
            if not hasattr(self.model, 'monitor'):
                # TODO: is this really necessary? I just put this error here
                # to prevent an AttributeError later, but I think we could
                # rewrite to avoid the AttributeError
                raise RuntimeError("The algorithm is responsible for setting"
                        " up the Monitor, but failed to.")
            if len(self.model.monitor._datasets)>0:
                # This monitoring channel keeps track of a shared variable,
                # which does not need inputs nor data.
                self.model.monitor.add_channel(name="monitor_seconds_per_epoch",
                                               ipt=None,
                                               val=self.monitor_time,
                                               data_specs=(NullSpace(), ''),
                                               dataset=self.model.monitor._datasets[0])
            self.run_callbacks_and_monitoring()
            while True:
                with log_timing(log, None, final_msg='Time this epoch:',
                                callbacks=[self.monitor_time.set_value]):
                    rval = self.algorithm.train(dataset=self.dataset)
                if rval is not None:
                    raise ValueError("TrainingAlgorithm.train should not return anything. Use TrainingAlgorithm.continue_learning to control whether learning continues.")
                self.model.monitor.report_epoch()
                self.run_callbacks_and_monitoring()
                if self.save_freq > 0 and self.model.monitor._epochs_seen % self.save_freq == 0:
                    self.save()
                continue_learning =  self.algorithm.continue_learning(self.model)
                assert continue_learning in [True, False, 0, 1]
                if not continue_learning:
                    break

        self.model.monitor.training_succeeded = True

        if self.save_freq > 0:
            self.save()

    def run_callbacks_and_monitoring(self):
        self.model.monitor()
        for extension in self.extensions:
            try:
                extension.on_monitor(self.model, self.dataset, self.algorithm)
            except TypeError, e:
                logging.warning('Failure during callback ' + str(extension))
                raise

    def save(self):
        """Saves the model."""
        #TODO-- save state of training algorithm so training can be
        # resumed after a crash
        for extension in self.extensions:
            extension.on_save(self.model, self.dataset, self.algorithm)
        if self.save_path is not None:
            with log_timing(log, 'Saving to ' + self.save_path):
                if self.first_save and (not self.allow_overwrite) \
                    and os.path.exists(self.save_path):
                    # Every job overwrites its own output on the second save
                    # and every save thereafter. The "allow_overwrite" flag
                    # only pertains to overwriting the output of previous jobs.
                    raise IOError("Trying to overwrite file when not allowed.")
                try:
                    # Make sure that saving does not serialize the dataset
                    self.dataset._serialization_guard = SerializationGuard()
                    serial.save(self.save_path, self.model,
                                on_overwrite='backup')
                finally:
                    self.dataset._serialization_guard = None
            self.first_save = False


class SerializationGuard(object):

    def __getstate__(self):
        raise IOError("You tried to serialize something that should not"
                " be serialized.")

if __name__ == "__main__":
    print >>sys.stderr, "ERROR: You probably meant to run scripts/train.py"
    sys.exit(1)
