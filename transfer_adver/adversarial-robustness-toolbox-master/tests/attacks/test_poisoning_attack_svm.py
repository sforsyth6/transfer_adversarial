# MIT License
#
# Copyright (C) IBM Corporation 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import numpy as np
from sklearn.svm import LinearSVC, NuSVC, SVC

from art.attacks import PoisoningAttackSVM
from art.classifiers import SklearnClassifier
from art.utils import master_seed, load_iris

logger = logging.getLogger('testLogger')

BATCH_SIZE = 10
NB_TRAIN = 10
NB_VALID = 10
NB_TEST = 10


class TestSVMAttack(unittest.TestCase):
    """
    A unittest class for testing Adversarial Patch attack.
    """

    @classmethod
    def setUpClass(cls):
        cls.setUpIRIS()

    @staticmethod
    def find_duplicates(x_train):
        """
        Returns an array of booleans that is true if that element was previously in the array

        :param x_train: training data
        :type x_train: `np.ndarray`
        :return: duplicates array
        :rtype: `np.ndarray`
        """
        dup = np.zeros(x_train.shape[0])
        for idx, x in enumerate(x_train):
            dup[idx] = np.isin(x_train[:idx], x).all(axis=1).any()
        return dup

    @classmethod
    def setUpIRIS(self):
        (x_train, y_train), (x_test, y_test), min_, max_ = load_iris()
        # Naturally IRIS has labels 0, 1, and 2. For binary classification use only classes 1 and 2.
        no_zero = np.where(np.argmax(y_train, axis=1) != 0)
        x_train = x_train[no_zero, :2][0]
        y_train = y_train[no_zero]
        no_zero = np.where(np.argmax(y_test, axis=1) != 0)
        x_test = x_test[no_zero, :2][0]
        y_test = y_test[no_zero]
        labels = np.zeros((y_train.shape[0], 2))
        labels[np.argmax(y_train, axis=1) == 2] = np.array([1, 0])
        labels[np.argmax(y_train, axis=1) == 1] = np.array([0, 1])
        y_train = labels
        te_labels = np.zeros((y_test.shape[0], 2))
        te_labels[np.argmax(y_test, axis=1) == 2] = np.array([1, 0])
        te_labels[np.argmax(y_test, axis=1) == 1] = np.array([0, 1])
        y_test = te_labels
        n_sample = len(x_train)

        order = np.random.permutation(n_sample)
        x_train = x_train[order]
        y_train = y_train[order].astype(np.float)

        x_train = x_train[:int(.9 * n_sample)]
        y_train = y_train[:int(.9 * n_sample)]
        train_dups = self.find_duplicates(x_train)
        x_train = x_train[np.logical_not(train_dups)]
        y_train = y_train[np.logical_not(train_dups)]
        test_dups = self.find_duplicates(x_test)
        x_test = x_test[np.logical_not(test_dups)]
        y_test = y_test[np.logical_not(test_dups)]
        self.iris = (x_train, y_train), (x_test, y_test), min_, max_

    def setUp(self):
        # Set master seed
        master_seed(1234)

    def test_linearSVC(self):
        """
        Test using a attack on LinearSVC
        """
        (x_train, y_train), (x_test, y_test), min_, max_ = self.iris

        # Build Scikitlearn Classifier
        clip_values = (min_, max_)
        clean = SklearnClassifier(model=LinearSVC(), clip_values=clip_values)
        clean.fit(x_train, y_train)
        poison = SklearnClassifier(model=LinearSVC(), clip_values=clip_values)
        poison.fit(x_train, y_train)
        attack = PoisoningAttackSVM(poison, 0.01, 1.0, x_train, y_train, x_test, y_test, 100)
        attack_point = attack.generate(np.array([x_train[0]]))
        poison.fit(x=np.vstack([x_train, attack_point]), y=np.vstack([y_train, np.copy(y_train[0].reshape((1, 2)))]))

        acc = np.average(np.all(clean.predict(x_test) == y_test, axis=1)) * 100
        poison_acc = np.average(np.all(poison.predict(x_test) == y_test, axis=1)) * 100
        logger.info("Clean Accuracy {}%".format(acc))
        logger.info("Poison Accuracy {}%".format(poison_acc))
        self.assertGreaterEqual(acc, poison_acc)

    def test_unsupported_kernel(self):
        (x_train, y_train), (x_test, y_test), min_, max_ = self.iris
        model = SVC(kernel='sigmoid')
        self.assertRaises(NotImplementedError, callable=PoisoningAttackSVM.__init__, classifier=model, step=0.01,
                          eps=1.0, x_train=x_train, y_train=y_train, x_val=x_test, y_val=y_test)

    def test_unsupported_SVC(self):
        (x_train, y_train), (x_test, y_test), _, _ = self.iris
        model = NuSVC()
        self.assertRaises(NotImplementedError, callable=PoisoningAttackSVM.__init__, classifier=model, step=0.01,
                          eps=1.0, x_train=x_train, y_train=y_train, x_val=x_test, y_val=y_test)

    def test_SVC_kernels(self):
        """
        First test with the TFClassifier.
        :return:
        """
        # Get MNIST
        (x_train, y_train), (x_test, y_test), min_, max_ = self.iris

        # Build Scikitlearn Classifier
        clip_values = (min_, max_)
        for kernel in ['linear', 'poly', 'rbf']:
            clean = SklearnClassifier(model=SVC(kernel=kernel), clip_values=clip_values)
            clean.fit(x_train, y_train)
            poison = SklearnClassifier(model=SVC(kernel=kernel), clip_values=clip_values)
            poison.fit(x_train, y_train)
            attack = PoisoningAttackSVM(poison, 0.01, 1.0, x_train, y_train, x_test, y_test, 100)
            attack_point = attack.generate(np.array([x_train[0]]))
            poison.fit(x=np.vstack([x_train, attack_point]),
                       y=np.vstack([y_train, np.copy(y_train[0].reshape((1, 2)))]))

            acc = np.average(np.all(clean.predict(x_test) == y_test, axis=1)) * 100
            poison_acc = np.average(np.all(poison.predict(x_test) == y_test, axis=1)) * 100
            logger.info("Clean Accuracy {}%".format(acc))
            logger.info("Poison Accuracy {}%".format(poison_acc))
            self.assertGreaterEqual(acc, poison_acc)


if __name__ == '__main__':
    unittest.main()
