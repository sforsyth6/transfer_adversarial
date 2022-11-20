import os
import sys

import numpy as np
import pickle

from sklearn.preprocessing import OneHotEncoder

class CIFAR100:

    def download_dataset(self, path, source='http://www.cs.toronto.edu/~kriz/'
                                      'cifar-100-python.tar.gz'):
        """
        Downloads and extracts the dataset, if needed.
        """
        files = ['train', 'test']
        for fn in files:
            if not os.path.exists(os.path.join(path, "cifar-100-python", fn)):
                break  # at least one file is missing
        else:
            return  # dataset is already complete

        print("Downloading and extracting %s into %s..." % (source, path))
        if sys.version_info[0] == 2:
            from urllib import urlopen
        else:
            from urllib.request import urlopen
        import tarfile
        if not os.path.exists(path):
            os.makedirs(path)
        u = urlopen(source)
        with tarfile.open(fileobj=u, mode='r|gz') as f:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(f, path=path)
        u.close()

    def load_dataset(self, path):
        self.download_dataset(path)

        # training data
        data = pickle.load(open(os.path.join(path, "cifar-100-python", "train"), 'rb'), encoding='latin1')
        X_train = data['data']
        y_train = np.asarray(data['fine_labels'], np.int8)

        # test data
        data = pickle.load(open(os.path.join(path, 'cifar-100-python', 'test'), 'rb'), encoding='latin1')
        X_test = data['data']
        y_test = np.asarray(data['fine_labels'], np.int8)

        # reshape
        X_train = X_train.reshape(-1, 3, 32, 32)
        X_test = X_test.reshape(-1, 3, 32, 32)

        # normalize
        try:
            mean_std = np.load(os.path.join(path, 'cifar-100-mean_std.npz'))
            mean = mean_std['mean']
            std = mean_std['std']
        except IOError:
            mean = X_train.mean(axis=(0,2,3), keepdims=True).astype(np.float32)
            std = X_train.std(axis=(0,2,3), keepdims=True).astype(np.float32)
            np.savez(os.path.join(path, 'cifar-100-mean_std.npz'),
                     mean=mean, std=std)
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std


        enc = OneHotEncoder(handle_unknown='ignore')

        enc.fit(y_train.reshape(-1, 1))

        y_train = enc.transform(y_train.reshape(-1, 1)).toarray()

        y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

        return (X_train, y_train), (X_test, y_test), 0, 1
