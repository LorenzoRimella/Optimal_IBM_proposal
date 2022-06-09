import numpy as np
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import sys
sys.path.append('AlternativeProposal/Scripts/')

from synthetic_data import *
from proposal_epidemic import *
from smc import *


N = 100
M = 2
covariates_n = 2

# Generate the covariates
W = np.random.normal(1, 0, (N, 1))
for i in range(covariates_n-1):

    cov = np.random.normal(0, 1, (N, 1))
    W = np.concatenate((W, cov), axis =1)

W           = tf.convert_to_tensor( W,                      dtype = tf.float32 )
beta_0      = tf.convert_to_tensor( [[-np.log(N-1)], [+0]], dtype = tf.float32 )  
beta_lambda = tf.convert_to_tensor( [[-1],           [+2]], dtype = tf.float32 ) 
beta_gamma  = tf.convert_to_tensor( [[-1],           [-1]], dtype = tf.float32 ) 

T     = 100
initial_distribution  = AB_initial(W, beta_0)
K_eta = AB_SIS_transition(W, beta_lambda, beta_gamma)
q     = tf.convert_to_tensor([[0.8], [0.8]], dtype = tf.float32)

Exp = compartmental_model(N, M, initial_distribution, K_eta, q)
C, Y = Exp.run(T)
while (tf.reduce_sum( tf.reduce_sum(C, axis =0), axis =1)[0]>(N*(T+1)-100)):
    C, Y = Exp.run(T)

# Whiteley-Rimella approximation
pi_0 = mean_SIS_initial(initial_distribution)
K_eta_mean = mean_SIS_transition(K_eta)

multi_approx = multinomial_approximation(N, M, pi_0, K_eta_mean, q)
pi, pi_T = multi_approx.run(Y)

# Proposal distribution
K_eta_multi = AB_SIS_transition_multi(W, beta_lambda, beta_gamma)

ESS_dict = {}

Nx = 512


BPF = proposal_BPF(Nx, N, M, initial_distribution, K_eta_multi, q,)
SMC_BPF = smc(BPF)
_, ESS_history, _ = SMC_BPF.run(Y)

ESS_NUMPY = ESS_history.numpy().reshape((1, T+1))

APF = proposal_APF(Nx, N, M, initial_distribution, K_eta_multi, q,)
SMC_APF = smc(APF)
_, ESS_history, _ = SMC_APF.run(Y)

ESS_NUMPY = np.concatenate((ESS_NUMPY, ESS_history.numpy().reshape((1, T+1))), axis = 0 ) 


h_list = [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 50]
for h in h_list:

    key = "alternative_"+str(h)
    
    proposal = alternative_proposal(Nx, N, M, initial_distribution, K_eta, K_eta_multi, q, h, pi_T)
    SMC_proposal = smc(proposal)
    _, ESS_history, _ = SMC_proposal.run(Y)

    ESS_NUMPY = np.concatenate((ESS_NUMPY, ESS_history.numpy().reshape((1, T+1))), axis = 0 ) 

np.save("AlternativeProposal/Data/Output/ESS/ESS_SIS_numpy", ESS_NUMPY)