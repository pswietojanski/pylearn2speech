import cPickle
import pickle
import numpy
import numpy as np
import os
import time
import warnings
import sys
import tables
from pylearn2.utils.string_utils import preprocess
from cPickle import BadPickleGet
io = None
hdf_reader = None
import struct
from pylearn2.utils import environ
from pylearn2.utils.string_utils import match
import shutil

def raise_cannot_open(path):
    pieces = path.split('/')
    for i in xrange(1,len(pieces)+1):
        so_far = '/'.join(pieces[0:i])
        if not os.path.exists(so_far):
            if i == 1:
                if so_far == '':
                    continue
                raise IOError('Cannot open '+path+' ('+so_far+' does not exist)')
            parent = '/'.join(pieces[0:i-1])
            bad = pieces[i-1]

            if not os.path.isdir(parent):
                raise IOError("Cannot open "+path+" because "+parent+" is not a directory.")

            candidates = os.listdir(parent)

            if len(candidates) == 0:
                raise IOError("Cannot open "+path+" because "+parent+" is empty.")

            if len(candidates) > 100:
                # Don't attempt to guess the right name if the directory is huge
                raise IOError("Cannot open "+path+" but can open "+parent+".")

            if os.path.islink(path):
                raise IOError(path + " appears to be a symlink to a non-existent file")
            raise IOError("Cannot open "+path+" but can open "+parent+". Did you mean "+match(bad,candidates)+" instead of "+bad+"?")
        # end if
    # end for
    assert False

def load(filepath, recurse_depth=0, retry = True):

    try:
        import joblib
        joblib_available = True
    except ImportError:
        joblib_available = False
    if recurse_depth == 0:
        filepath = preprocess(filepath)

    if filepath.endswith('.npy'):
        return np.load(filepath)

    if filepath.endswith('.mat'):
        global io
        if io is None:
            import scipy.io
            io = scipy.io
        try:
            return io.loadmat(filepath)
        except NotImplementedError, nei:
            if str(nei).find('HDF reader') != -1:
                global hdf_reader
                if hdf_reader is None:
                    import h5py
                    hdf_reader = h5py
                return hdf_reader.File(filepath)
            else:
                raise
        #this code should never be reached
        assert False

    def exponential_backoff():
        if recurse_depth > 9:
            print ('Max number of tries exceeded while trying to open ' +
                   filepath)
            print 'attempting to open via reading string'
            f = open(filepath, 'rb')
            lines = f.readlines()
            f.close()
            content = ''.join(lines)
            return cPickle.loads(content)
        else:
            nsec = 0.5 * (2.0 ** float(recurse_depth))
            print "Waiting " + str(nsec) + " seconds and trying again"
            time.sleep(nsec)
            return load(filepath, recurse_depth + 1, retry)

    try:
        if not joblib_available:
            with open(filepath, 'rb') as f:
                obj = cPickle.load(f)
        else:
            try:
                obj = joblib.load(filepath)
            except Exception, e:
                if os.path.exists(filepath) and not os.path.isdir(filepath):
                    raise
                raise_cannot_open(filepath)


    except BadPickleGet, e:
        print ('Failed to open ' + str(filepath) +
               ' due to BadPickleGet with exception string ' + str(e))

        if not retry:
            raise
        obj =  exponential_backoff()
    except EOFError, e:

        print ('Failed to open ' + str(filepath) +
               ' due to EOFError with exception string ' + str(e))

        if not retry:
            raise
        obj =  exponential_backoff()
    except ValueError, e:
        print ('Failed to open ' + str(filepath) +
               ' due to ValueError with string ' + str(e))

        if not retry:
            raise
        obj =  exponential_backoff()
    except Exception, e:
        #assert False
        exc_str = str(e)
        if len(exc_str) > 0:
            import pdb
            tb = pdb.traceback.format_exc()
            raise Exception("Couldn't open '" + str(filepath) +
                            "' due to: " + str(type(e)) + ', ' + str(e) +
                            ". Orig traceback:\n" + tb)
        else:
            print ("Couldn't open '" + str(filepath) +
                   "' and exception has no string. Opening it again outside "
                   "the try/catch so you can see whatever error it prints on "
                   "its own.")
            f = open(filepath, 'rb')
            obj = cPickle.load(f)
            f.close()

    #if the object has no yaml_src, we give it one that just says it
    #came from this file. could cause trouble if you save obj again
    #to a different location
    if not hasattr(obj,'yaml_src'):
        try:
            obj.yaml_src = '!pkl: "'+os.path.abspath(filepath)+'"'
        except:
            pass

    return obj


def save(filepath, obj, on_overwrite = 'ignore'):
    """
    Serialize `object` to a file denoted by `filepath`.

    Parameters
    ----------
    filepath : str
        A filename. If the suffix is `.joblib` and joblib can be
        imported, `joblib.dump` is used in place of the regular
        pickling mechanisms; this results in much faster saves by
        saving arrays as separate .npy files on disk. If the file
        suffix is `.npy` than `numpy.save` is attempted on `obj`.
        Otherwise, (c)pickle is used.

    obj : object
        A Python object to be serialized.

    on_overwrite: A string specifying what to do if the file already
                exists.
                ignore: just overwrite it
                backup: make a copy of the file (<filepath>.bak) and
                        delete it when done saving the new copy.
                        this allows recovery of the old version of
                        the file if saving the new one fails
    """


    filepath = preprocess(filepath)

    if os.path.exists(filepath):
        if on_overwrite == 'backup':
            backup = filepath + '.bak'
            shutil.move(filepath, backup)
            save(filepath, obj)
            try:
                os.remove(backup)
            except Exception, e:
                warnings.warn("Got an error while traing to remove "+backup+":"+str(e))
            return
        else:
            assert on_overwrite == 'ignore'


    try:
        _save(filepath, obj)
    except RuntimeError, e:
        """ Sometimes for large theano graphs, pickle/cPickle exceed the
            maximum recursion depth. This seems to me like a fundamental
            design flaw in pickle/cPickle. The workaround I employ here
            is the one recommended to someone who had a similar problem
            on stackexchange:

            http://stackoverflow.com/questions/2134706/hitting-maximum-recursion-depth-using-pythons-pickle-cpickle

            Obviously this does not scale and could cause a crash
            but I don't see another solution short of writing our
            own implementation of pickle.
        """
        if str(e).find('recursion') != -1:
            warnings.warn('pylearn2.utils.save encountered the following '
                          'error: ' + str(e) +
                          '\nAttempting to resolve this error by calling ' +
                          'sys.setrecusionlimit and retrying')
            old_limit = sys.getrecursionlimit()
            try:
                sys.setrecursionlimit(50000)
                _save(filepath, obj)
            finally:
                sys.setrecursionlimit(old_limit)


def get_pickle_protocol():
    """
    Allow configuration of the pickle protocol on a per-machine basis.
    This way, if you use multiple platforms with different versions of
    pickle, you can configure each of them to use the highest protocol
    supported by all of the machines that you want to be able to
    communicate.
    """
    try:
        protocol_str = os.environ['PYLEARN2_PICKLE_PROTOCOL']
    except KeyError:
        # If not defined, we default to 0 because this is the default
        # protocol used by cPickle.dump (and because it results in
        # maximum portability)
        protocol_str = '0'
    if protocol_str == 'pickle.HIGHEST_PROTOCOL':
        return pickle.HIGHEST_PROTOCOL
    return int(protocol_str)

def _save(filepath, obj):
    try:
        import joblib
        joblib_available = True
    except ImportError:
        joblib_available = False
    if filepath.endswith('.npy'):
        np.save(filepath, obj)
        return
    # This is dumb
    # assert filepath.endswith('.pkl')
    save_dir = os.path.dirname(filepath)
    # Handle current working directory case.
    if save_dir == '':
        save_dir = '.'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if os.path.exists(save_dir) and not os.path.isdir(save_dir):
        raise IOError("save path %s exists, not a directory" % save_dir)
    elif not os.access(save_dir, os.W_OK):
        raise IOError("permission error creating %s" % filepath)
    try:
        if joblib_available and filepath.endswith('.joblib'):
            joblib.dump(obj, filepath)
        else:
            if filepath.endswith('.joblib'):
                warnings.warn('Warning: .joblib suffix specified but joblib '
                              'unavailable. Using ordinary pickle.')
            with open(filepath, 'wb') as filehandle:
                cPickle.dump(obj, filehandle, get_pickle_protocol())
    except Exception, e:
        # TODO: logging, or warning
        print "cPickle has failed to write an object to " + filepath
        if str(e).find('maximum recursion depth exceeded') != -1:
            raise
        try:
            # TODO: logging, or warning
            print 'retrying with pickle'
            with open(filepath, "wb") as f:
                pickle.dump(obj, f)
        except Exception, e2:
            if str(e) == '' and str(e2) == '':
                # TODO: logging, or warning
                print (
                    'neither cPickle nor pickle could write to %s' % filepath
                )
                print (
                    'moreover, neither of them raised an exception that '
                    'can be converted to a string'
                )
                print (
                    'now re-attempting to write with cPickle outside the '
                    'try/catch loop so you can see if it prints anything '
                    'when it dies'
                )
                with open(filepath, 'wb') as f:
                    cPickle.dump(obj, f, get_pickle_protocol())
                print ('Somehow or other, the file write worked once '
                       'we quit using the try/catch.')
            else:
                if str(e2) == 'env':
                    raise

                import pdb
                tb = pdb.traceback.format_exc()
                raise IOError(str(obj) +
                              ' could not be written to '+
                              str(filepath) +
                              ' by cPickle due to ' + str(e) +
                              ' nor by pickle due to ' + str(e2) +
                              '. \nTraceback '+ tb)
        print ('Warning: ' + str(filepath) +
               ' was written by pickle instead of cPickle, due to '
               + str(e) +
               ' (perhaps your object is really big?)')


def clone_via_serialize(obj):
    s = cPickle.dumps(obj, get_pickle_protocol())
    return cPickle.loads(s)


def to_string(obj):
    return cPickle.dumps(obj, get_pickle_protocol())

def from_string(s):
    return cPickle.loads(s)


def mkdir(filepath):
    """
    Make a directory. Should succeed even if it needs to make more than one
    directory and nest subdirectories to do so. Raises an error if the
    directory can't be made. Does not raise an error if the directory
    already exists.
    """
    try:
        os.makedirs(filepath)
    except:
        if not os.path.isdir(filepath):
            raise

def read_int( fin, n = 1):
    if n == 1:
        s = fin.read(4)
        if len(s) != 4:
            raise ValueError('fin did not contain 4 bytes')
        return struct.unpack('i', s)[0]
    else:
        rval = []
        for i in xrange(n):
            rval.append(read_int(fin))
        return rval

#dictionary to convert lush binary matrix magic numbers
#to dtypes
lush_magic = {
            507333717 : 'uint8',
            507333716 : 'int32',
            507333713 : 'float32',
            507333715 : 'float64'
        }

def read_bin_lush_matrix(filepath):
    f = open(filepath,'rb')
    try:
        magic = read_int(f)
    except ValueError:
        raise ValueError("Couldn't read magic number")
    ndim = read_int(f)

    if ndim == 0:
        shape = ()
    else:
        shape = read_int(f, max(3, ndim))

    total_elems = 1
    for dim in shape:
        total_elems *= dim

    try:
        dtype = lush_magic[magic]
    except KeyError:
        raise ValueError('Unrecognized lush magic number '+str(magic))

    rval = np.fromfile(file = f, dtype = dtype, count = total_elems)

    excess = f.read(-1)

    if excess != '':
        raise ValueError(str(len(excess))+' extra bytes found at end of file.'
                ' This indicates  mismatch between header and content')

    rval = rval.reshape(*shape)

    f.close()

    return rval

def load_train_file(config_file_path):
    """Loads and parses a yaml file for a Train object.
    Publishes the relevant training environment variables"""
    from pylearn2.config import yaml_parse

    suffix_to_strip = '.yaml'

    # publish environment variables related to file name
    if config_file_path.endswith(suffix_to_strip):
        config_file_full_stem = config_file_path[0:-len(suffix_to_strip)]
    else:
        config_file_full_stem = config_file_path

    for varname in ["PYLEARN2_TRAIN_FILE_NAME", #this one is deprecated
            "PYLEARN2_TRAIN_FILE_FULL_STEM"]: #this is the new, accepted name
        environ.putenv(varname, config_file_full_stem)

    directory = config_file_path.split('/')[:-1]
    directory = '/'.join(directory)
    if directory != '':
        directory += '/'
    environ.putenv("PYLEARN2_TRAIN_DIR", directory)
    environ.putenv("PYLEARN2_TRAIN_BASE_NAME", config_file_path.split('/')[-1] )
    environ.putenv("PYLEARN2_TRAIN_FILE_STEM", config_file_full_stem.split('/')[-1] )

    return yaml_parse.load_path(config_file_path)


def save_params_to_pytables(filepath, params, container_name='params', desc='', on_overwrite='ignore', filters=None, params_symbolic=True):
    """Makes an hdf5 file with params
    path: filename of the param dump
    params: dictionary with {'param_name': value}
    desc: optional comment to save in file
    filters: use this, to specify different than default options to tables
    params_symbolic: if True, means they are Theano variables
    """

    def save_pytables_symbolic(filepath, params, container_name, desc, filters):
        if filters is None:
            filters = tables.Filters(complib='zlib', complevel=1)
        with tables.open_file(filepath, mode = "w", \
                              title = "Model parameters: %s"%filepath) as h5file:
            gcolumns = h5file.create_group(h5file.root, container_name, desc)

            for param in params:
                p_value = param.get_value()
                p_atom = tables.Atom.from_dtype(p_value.dtype)
                p_array = h5file.createCArray(gcolumns, param.name, atom = p_atom, shape = p_value.shape,
                                title = param.name, filters = filters)
                p_array[:] = p_value
                h5file.flush()

                print 'ModelPyTables: exporting param %s with shape %s and dtype %s'%(param.name, p_value.shape, p_value.dtype)

    def save_pytables_raw(filepath, params, container_name, desc, filters):
        if filters is None:
            filters = tables.Filters(complib='zlib', complevel=1)
        with tables.open_file(filepath, mode = "w", \
                              title = "Model parameters: %s"%filepath) as h5file:
            gcolumns = h5file.create_group(h5file.root, container_name, desc)

            for p_name, p_value in params.iteritems():
                p_atom = tables.Atom.from_dtype(p_value.dtype)
                p_array = h5file.createCArray(gcolumns, p_name, atom=p_atom, shape=p_value.shape,
                                title=p_name, filters=filters)
                p_array[:] = p_value
                h5file.flush()

                print 'ModelPyTables: exporting param %s with shape %s and dtype %s'%(p_name, p_value.shape, p_value.dtype)

    filepath = preprocess(filepath)

    if os.path.exists(filepath):
        if on_overwrite == 'backup':
            backup = filepath + '.bak'
            shutil.move(filepath, backup)
            if params_symbolic:
                save_pytables_symbolic(filepath, params, container_name, desc, filters)
            else:
                save_pytables_raw(filepath, params, container_name, desc, filters)
            try:
                os.remove(backup)
            except Exception, e:
                warnings.warn("Got an error while traing to remove "+backup+":"+str(e))
            return
        else:
            assert on_overwrite == 'ignore'

    if params_symbolic:
        save_pytables_symbolic(filepath, params, container_name, desc, filters)
    else:
        save_pytables_raw(filepath, params, container_name, desc, filters)


def load_params_from_pytables(filepath, container_name=None):
    """Returns dictionary {'param_name': value} so the model can appropriately set those parameters back.
    Bear in mind this is just a dictionary of ndarrays. These should be then loaded into an appropriate Theano variables.
    The advantage is theano variables could be easily build for desired backend (GPU, CPU) first"""

    params = {}
    with tables.open_file(filepath, mode = "r") as h5file:
        container_names = []
        if container_name is None:
            for group in h5file.walk_groups('/'):
                container_names.append(group._v_name)
        else:
            container_names.append(container_name)

        for container_name in container_names:
            if container_name == '/': continue
            for node in h5file.walk_nodes('/%s'%container_name, "Array"):
                if params.has_key(node._v_name):
                    raise KeyError('Key already exists %s'%node.name) #it should not happen, but check anyway
                params[node._v_name] = node.read()
            #print 'ModelPyTables: Lodaing param %s into dictionary (shape is %s and dtype %s)'%\
            #                                   (node.name, params[node.name].shape, params[node.name].dtype)

    return params


def export_weights_to_kaldi(kaldi_nnet_file, pytables_file, container_name=None):
    """
    This function is full of hacks and written for single experiment only
    """

    params = load_params_from_pytables(pytables_file, container_name)
    num_layers = len(params) / 2
    with open(kaldi_nnet_file, 'w') as f:

        for i in xrange(0, num_layers - 1):

            print 'Exporting h%d_W' % i
            W = params['h%d_W' % i]
            print 'Exporting h%d_b' % i
            b = params['h%d_b' % i]

            print >> f, '<affinetransform>', W.shape[1], W.shape[0]
            print >> f, '[ ',
            for i in xrange(W.shape[1]):
                for j in xrange(W.shape[0]):
                    f.write('%f ' % W[j,i])
                if i != (W.shape[1]-1):
                    print >> f, ' '
            print >> f, ']'
            #bias vector
            print >> f, '[ ',
            for i in xrange(b.shape[0]):
                f.write('%f ' % b[i])
            print >> f, ']'
            print >> f, '<sigmoid>', W.shape[1],  W.shape[1]

        #logistic regression + softmax
        W = params['softmax_W']
        b = params['softmax_b']

        print >> f, '<affinetransform>', W.shape[1], W.shape[0]
        print >> f, '[ ',
        for i in xrange(W.shape[1]):
            for j in xrange(W.shape[0]):
                f.write('%f ' % W[j, i])
            if i!=(W.shape[1]-1): print >> f, ' '
        print >> f, ']'
        #bias vector
        print >> f, '[ ',
        for i in xrange(b.shape[0]):
            f.write('%f '%b[i])
        print >> f, ']'
        print >> f, '<softmax>', W.shape[1], W.shape[1]

    return None


def kaldi_to_pytables(kaldi_nnet_file, pytables_file=None, container_name=None):
    """
    This function is full of hacks and written for single experiment only
    """
    from collections import OrderedDict
    from pylearn2.datasets.speech_utils.kaldi_providers import read_kaldi_matrix, read_uttid

    def get_affine(f):
        p1, rows = struct.unpack('<bi', f.read(5))
        p2, cols = struct.unpack('<bi', f.read(5))
        t1= read_uttid(f)
        __, lrcoef = struct.unpack('<bf', f.read(5))
        t2= read_uttid(f)
        __, bcoef = struct.unpack('<bf', f.read(5))
        t3= read_uttid(f)
        __, mnorm = struct.unpack('<bf', f.read(5))
        #print t1, t2, t3, lrcoef, bcoef, mnorm
        W = read_kaldi_matrix(f, skip_binary_preamble=True)
        b = read_kaldi_matrix(f, skip_binary_preamble=True)
        return W.T, b #kaldi has different orientation

    tokens = ['<AffineTransform>', '<Softmax>', '<Sigmoid>', '<Nnet>',
              '</Nnet>', '<LearnRateCoef>', '<BiasLearnRateCoef>',
              '<MaxNorm>', '<Splice>', '<AddShift>', '<Rescale>']

    layers = []
    with open(kaldi_nnet_file, 'rb') as f:

        mode = struct.unpack('<xc', f.read(2))
        if mode[0] != "B":
            raise ValueError('Only binary loading is supported, got %s'%mode[0])

        token = read_uttid(f)
        assert token == "<Nnet>", (
            'Got %s'%token
        )

        while f.read(1):
            f.seek(-1, 1)

            token = read_uttid(f)
            assert token in tokens, (
                "Expected token, got %s"%token
            )

            if token == "<AffineTransform>":
                W, b = get_affine(f)
                #print 'Loaded weights and biases, shapes are:', W.shape, b.shape
                layers.append(OrderedDict([('W', W), ('b', b)]))
            elif token == "<Sigmoid>":
                __, dim1 = struct.unpack('<bi', f.read(5))
                __, dim2 = struct.unpack('<bi', f.read(5))
                assert dim1 == dim2, (
                    "Dims should be the same for sigmoid"
                )
                #print 'Loaded Sigmoid non-linearity with dim', dim1
                #layers.append('Sigmoid')
            elif token == '<Softmax>':
                __, dim1 = struct.unpack('<bi', f.read(5))
                __, dim2 = struct.unpack('<bi', f.read(5))
                assert dim1 == dim2, (
                    "Dims should be the same for softmax"
                )
                #print 'Loaded Softmax non-linearity with dim', dim1
                #layers.append('Softmax')
            elif token == '<Splice>':
                __, dim1 = struct.unpack('<bi', f.read(5))
                __, dim2 = struct.unpack('<bi', f.read(5))
                #print 'Loaded splice component with', splice
                splice = read_kaldi_matrix(f, skip_binary_preamble=True)
                #can skip this
            elif token == '<AddShift>':
                __, dim1 = struct.unpack('<bi', f.read(5))
                __, dim2 = struct.unpack('<bi', f.read(5))
                t1= read_uttid(f)
                __, lrcoef = struct.unpack('<bf', f.read(5))
                shift = read_kaldi_matrix(f, skip_binary_preamble=True)
                #print 'Loaded AddShift component with dims', dim1, dim2
                layers.append(OrderedDict([('shift', shift)]))
            elif token == '<Rescale>':
                __, dim1 = struct.unpack('<bi', f.read(5))
                __, dim2 = struct.unpack('<bi', f.read(5))
                t1= read_uttid(f)
                __, lrcoef = struct.unpack('<bf', f.read(5))
                scale = read_kaldi_matrix(f, skip_binary_preamble=True)
                #print 'Loaded Rescale component with dims', dim1, dim2
                layers.append(OrderedDict([('scale', scale)]))
            elif token == "</Nnet>":
                #print 'Loaded all componentes!'
                break;
            else:
                raise NotImplementedError('Token %s not yet supported' % token)

    assert token == '</Nnet>', (
        'Looks like loaded model is incomplete as the last '
        'token is %s (should be </Nnet>)' % token
    )

    #in case file exists, load it and merge
    params={}
    if pytables_file is not None and os.path.isfile(pytables_file):
        print 'Loading %s first (possibly to merge params)'
        params = load_params_from_pytables(pytables_file)

    #do the mapping
    for i, param in enumerate(layers):
        for key, value in param.iteritems():
            prefix = 'h'
            if key not in ['W', 'b']:
                prefix = 'g'

            if i < len(layers) - 1 or key not in ['W', 'b']:
                keyf = '%s%d_%s' % (prefix, i, key)
            else: #softmax otherwise
                keyf = 'softmax_%s' % (key)

            #if keyf in params:
                #print 'The key exists, will override the param %s' % keyf
            params[keyf] = value

    if pytables_file is not None:
        print 'Saving params %s to hdf container in %s' % (params.keys(), pytables_file)
        save_params_to_pytables(pytables_file, params, params_symbolic=False)

    return params
