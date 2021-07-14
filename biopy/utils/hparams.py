from collections import MutableMapping
from warnings import warn

import warnings
warnings.simplefilter('always', UserWarning)

#Adapted from https://stackoverflow.com/a/47086935
class HParams(MutableMapping):
    def __init__(self, **kwargs):
        self._lookup = {}
        self.data = {}
        if kwargs:
            self.data.update(kwargs)

    def __getitem__(self, key):
        retval = self.data[key]
        self._lookup[key] = self._lookup.get(key, 0) + 1
        return retval
        
    def get(self, key, default):
        if key not in self.data:
            warn(f'Hyperparameter {key} is not set, setting default value of {default}', UserWarning, stacklevel=2)
            self.data[key] = default
            
        return self[key]
        
    def __getattr__(self, key):
        return self[key]

    def __setitem__(self, key, value):
        if key in self.data and self.data[key] != value:
            warn(f'Hyperparameter {key} is already present with a value of {self.data[key]}, setting new value of {value}', UserWarning, stacklevel=2)
            self._lookup[key] = 0
        self.data[key] = value

    def __delitem__(self, key):
        warn(f'Hyperparameter {key} is being deleted from hyperparameters object', UserWarning, stacklevel=2)
        del self.data[key]
        self._lookup[key] = 0
        del self._lookup[key]

    def items(self):
        #warn('item is being called on hyperparameters object, not going to count key accesses', UserWarning, stacklevel=2)
        yield from self.data.items()

    def __iter__(self):
        #warn('__iter__ is being called on hyperparameters object, not going to count key accesses', UserWarning, stacklevel=2)
        yield from self.data

    def __len__(self):
        return len(self.data)
        
    def _get_unused_keys(self):
        return [key for key in self.data if self._lookup.get(key, 0) == 0]
        
    def warn_if_unused_keys(self):
        for key in self._get_unused_keys():
            warn(f'Hyperparameter {key} is set but never read so far, typo?', UserWarning, stacklevel=2)

    def raise_if_unused_keys(self):
        unused_keys = self._get_unused_keys()
        if len(unused_keys) > 0:
            raise RuntimeError('Hyperparameters object has got some unused hyperparameters: ' + ', '.join(map(str, unused_keys)))
