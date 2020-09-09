"""
author: stefan.depperschmidt@gmail.com

plainhmm.py
"""

import numpy as np
from hmmlearn.hmm import GaussianHMM


class Hmm:
    '''
    this instance is useful for training hmm with gaussians
    use
    '''

    def __init__(self, nparray, number_of_states):

        # nparray input
        self.nparray = nparray

        # number of states to infer
        self.number_of_states = number_of_states

        # init model with predefined number of states
        self.model = GaussianHMM(self.number_of_states)

        # train model
        self.model.fit(self.nparray)

        # states inferred from training data
        self.states = self.model.predict(self.nparray)

    # predict method
    def predict(self, test_nparray):

        predicted_states = self.model.predict(test_nparray)

        return predicted_states
