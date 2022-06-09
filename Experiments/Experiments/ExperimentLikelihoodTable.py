import numpy as np
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import sys
sys.path.append('AlternativeProposal/Scripts/')

from synthetic_data import *
from proposal_epidemic import *
from smc import *

import time

########################################
# Generate data
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

########################################
# Whiteley-Rimella approximation
pi_0 = mean_SIS_initial(initial_distribution)
K_eta_mean = mean_SIS_transition(K_eta)

multi_approx = multinomial_approximation(N, M, pi_0, K_eta_mean, q)
pi, pi_T = multi_approx.run(Y)


########################################
########################################
# likelihood experiment
replicates = 100
Nx_list    = [64, 128, 256, 512, 1024, 2048]
h_list     = [0, 1, 2, 5, 10, 20]

loglikelihood_DGP    = np.zeros((len(Nx_list), len(h_list)+1, replicates))
loglikelihood_NotDGP = np.zeros((len(Nx_list), len(h_list)+1, replicates))

cost_DGP    = np.zeros((len(Nx_list), len(h_list)+1, replicates))
cost_NotDGP = np.zeros((len(Nx_list), len(h_list)+1, replicates))

K_eta_multi = AB_SIS_transition_multi(W, beta_lambda, beta_gamma)

# DGP
i = -1
for Nx in Nx_list:

    i = i +1
    for k in range(replicates):
        APF = proposal_APF(Nx, N, M, initial_distribution, K_eta_multi, q,)
        SMC_APF = smc(APF)

        startTime = time.time()
        _, _, loglikelihood_APF = SMC_APF.run(Y)
        executionTime = (time.time() - startTime)

        j = 0
        loglikelihood_DGP[i, j, k] = tf.reduce_sum(loglikelihood_APF).numpy()
        cost_DGP[i, j, k]          = executionTime

        for h in h_list:
            j = j +1

            proposal = alternative_proposal(Nx, N, M, initial_distribution, K_eta, K_eta_multi, q, h, pi_T)
            SMC_proposal = smc(proposal)

            startTime = time.time()
            _, _, loglikelihood = SMC_proposal.run(Y)
            executionTime = (time.time() - startTime)            

            loglikelihood_DGP[i, j, k] = (tf.reduce_sum(loglikelihood).numpy())
            cost_DGP[i, j, k]          = executionTime


# generate wrong parameter
beta_lambda = tf.convert_to_tensor( [[-3],           [+0]], dtype = tf.float32 ) 

initial_distribution  = AB_initial(W, beta_0)
K_eta = AB_SIS_transition(W, beta_lambda, beta_gamma)

# Whiteley-Rimella approximation
pi_0 = mean_SIS_initial(initial_distribution)
K_eta_mean = mean_SIS_transition(K_eta)

multi_approx = multinomial_approximation(N, M, pi_0, K_eta_mean, q)
pi, pi_T = multi_approx.run(Y)

K_eta_multi = AB_SIS_transition_multi(W, beta_lambda, beta_gamma)

i = -1
for Nx in Nx_list:

    i = i +1
    for k in range(replicates):
        APF = proposal_APF(Nx, N, M, initial_distribution, K_eta_multi, q,)
        SMC_APF = smc(APF)

        startTime = time.time()
        _, _, loglikelihood_APF = SMC_APF.run(Y)
        executionTime = (time.time() - startTime)

        j = 0
        loglikelihood_NotDGP[i, j, k] = tf.reduce_sum(loglikelihood_APF).numpy()
        cost_NotDGP[i, j, k]          = executionTime

        for h in h_list:
            j = j +1

            proposal = alternative_proposal(Nx, N, M, initial_distribution, K_eta, K_eta_multi, q, h, pi_T)
            SMC_proposal = smc(proposal)
            startTime = time.time()
            _, _, loglikelihood = SMC_proposal.run(Y)
            executionTime = (time.time() - startTime)            

            loglikelihood_NotDGP[i, j, k] = (tf.reduce_sum(loglikelihood).numpy())
            cost_NotDGP[i, j, k]          = executionTime


np.save("AlternativeProposal/Data/Output/LikelihoodTable/LikelihoodTable_SIS_DGP_numpy",    loglikelihood_DGP)
np.save("AlternativeProposal/Data/Output/LikelihoodTable/CostTable_SIS_DGP_numpy",          cost_DGP)
np.save("AlternativeProposal/Data/Output/LikelihoodTable/LikelihoodTable_SIS_NotDGP_numpy", loglikelihood_NotDGP)
np.save("AlternativeProposal/Data/Output/LikelihoodTable/CostTable_SIS_NotDGP_numpy",       cost_NotDGP)
