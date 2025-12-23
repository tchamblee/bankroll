import os
import numpy as np

class LazyMMapContext:
    """
    Acts like a dictionary but lazy-loads numpy arrays from a directory using mmap.
    This prevents loading the entire dataset into RAM and allows efficient parallel access.
    """
    def __init__(self, directory):
        self.directory = directory
        self.cache = {}
        # Pre-scan directory to know available keys
        self.available_keys = {f.replace('.npy', '') for f in os.listdir(directory) if f.endswith('.npy')}

    def __getitem__(self, key):
        if key in self.cache:
            return self.cache[key]
            
        if key not in self.available_keys:
            # Check if it's special key like __len__
            if key == '__len__':
                # Try to infer length from any file
                if not self.available_keys: return 0
                first_key = next(iter(self.available_keys))
                arr = self[first_key]
                return len(arr)
            raise KeyError(f"Feature '{key}' not found in context directory.")

        path = os.path.join(self.directory, f"{key}.npy")
        # Load in mmap mode 'r' (Read-Only, Shared)
        arr = np.load(path, mmap_mode='r')
        self.cache[key] = arr
        return arr

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default
            
    def __contains__(self, key):
        return key in self.cache or key in self.available_keys

    def __len__(self):
        return self['__len__']

    def keys(self):
        return self.available_keys

    def values(self):
        for k in self.available_keys:
            yield self[k]
