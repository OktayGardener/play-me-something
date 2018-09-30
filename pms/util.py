import functools
import glob
import subprocess
import re

import tensorflow as tf


def generic_glob(pathname):
    """Glob that works both on gcs and local disk
    :return A list of matching filenames
    """
    if pathname.startswith('gs://'):
        return subprocess.check_output(['gsutil', 'ls', pathname]).split()
    else:
        return glob.glob(pathname)


def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
            print('%s: %s' % (function.__name__, getattr(self, attribute)))
        return getattr(self, attribute)

    return wrapper


_lazy_vals = {}


def monitor_tensor(name, tensor):
    print('* %s: %s%s' % (name, tensor.dtype.name, tensor.shape.as_list()))
    tf.summary.histogram(name, tensor)


def cached(function):
    @functools.wraps(function)
    def wrapper():
        if function not in _lazy_vals:
            _lazy_vals[function] = function()
        return _lazy_vals[function]

    return wrapper


def get_spotify_username(spotify_username):
    command = "jhurl -s spotify.internal.stuff username=%s" % spotify_username
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=None, shell=True)

    # Launch the shell command:
    output = process.communicate()
    match = re.search(r"(\"user_id\":\")([a-z0-9]*)(?=\")", output[0])
    if match:
        return match.group(2)
    else:
        return None
