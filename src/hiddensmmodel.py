"""
author: stefan.depperschmidt@gmail.com

hiddensmmodel.py
"""

import numpy as np
import pyhsmm
from pyhsmm.util.text import progprint_xrange


class Hsmm:
    '''
    this instance is useful for training an hdp-hsmm
    use 
    '''
    
    def __init__(self, nparray):
        
        # nparray input
        self.nparray = nparray
        
        # get observation dimension
        self.obs_dim = nparray.shape[1]
        
        # define emission parameters
        self.obs_hypparams = {'mu_0': np.zeros(self.obs_dim),
                              'sigma_0': np.eye(self.obs_dim),
                              'kappa_0': 0.25,
                              'nu_0': self.obs_dim+2}
        
        # define duration parameters
        self.dur_hypparams = {'alpha_0':2*30,
                         'beta_0':2}
        
        # define max hidden states to uncover
        self.Nmax = 10
        
        # init emission distributions
        self.obs_distns = [pyhsmm.distributions.Gaussian(**self.obs_hypparams) 
                           for state in range(self.Nmax)]
        
        # init duration distributions
        self.dur_distns = [pyhsmm.distributions.PoissonDuration(
                           **self.dur_hypparams) for state in range(self.Nmax)]
        # init hdp-hsmm
        self.posteriormodel = pyhsmm.models.WeakLimitHDPHSMM(alpha=6.,gamma=6.,
                                                   init_state_concentration=6.,
                                                    obs_distns=self.obs_distns,
                                                    dur_distns=self.dur_distns)
        
        # add data 
        self.posteriormodel.add_data(self.nparray, trunc=60)
        
        # train the model
        for idx in progprint_xrange(200):
            self.posteriormodel.resample_model()

        # results
        self.states = self.posteriormodel.stateseqs
