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

T     = 100 

q_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

replicates = 100
Nx         = 512
h_list     = [0, 1, 2, 5, 10, 20]

loglikelihood_DGP    = np.zeros((len(q_list), len(h_list)+1, replicates))
loglikelihood_NotDGP = np.zeros((len(q_list), len(h_list)+1, replicates))

ESS_DGP    = np.zeros((len(q_list), len(h_list)+1, replicates, T+1))
ESS_NotDGP = np.zeros((len(q_list), len(h_list)+1, replicates, T+1))

i = -1
for q_value in q_list:

    i = i +1
    ########################################
    # Generate data
    N = 100
    M = 2
    covariates_n = 2

    # Generate the covariates
    W = np.random.normal(1, 0, (N, 1))
    for c in range(covariates_n-1):

        cov = np.random.normal(0, 1, (N, 1))
        W = np.concatenate((W, cov), axis =1)

    W           = tf.convert_to_tensor( W,                      dtype = tf.float32 )
    beta_0      = tf.convert_to_tensor( [[-np.log(N-1)], [+0]], dtype = tf.float32 )  
    beta_lambda = tf.convert_to_tensor( [[-1],           [+2]], dtype = tf.float32 ) 
    beta_gamma  = tf.convert_to_tensor( [[-1],           [-1]], dtype = tf.float32 ) 

    initial_distribution  = AB_initial(W, beta_0)
    K_eta = AB_SIS_transition(W, beta_lambda, beta_gamma)
    q     = tf.convert_to_tensor([[q_value], [q_value]], dtype = tf.float32)

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

    K_eta_multi = AB_SIS_transition_multi(W, beta_lambda, beta_gamma)

    for k in range(replicates):
        APF = proposal_APF(Nx, N, M, initial_distribution, K_eta_multi, q,)
        SMC_APF = smc(APF)

        _, ESS_history, loglikelihood_APF = SMC_APF.run(Y)

        j = 0
        loglikelihood_DGP[i, j, k] = tf.reduce_sum(loglikelihood_APF).numpy()
        ESS_DGP[i, j, k, :]        = ESS_history

        for h in h_list:
            j = j +1

            proposal = alternative_proposal(Nx, N, M, initial_distribution, K_eta, K_eta_multi, q, h, pi_T)
            SMC_proposal = smc(proposal)

            startTime = time.time()
            _, ESS_history, loglikelihood = SMC_proposal.run(Y)
            executionTime = (time.time() - startTime)            

            loglikelihood_DGP[i, j, k] = (tf.reduce_sum(loglikelihood).numpy())
            ESS_DGP[i, j, k, :]        = ESS_history



########################################
# Generate data
N = 100
M = 2
covariates_n = 2

# Generate the covariates
W = np.random.normal(1, 0, (N, 1))
for c in range(covariates_n-1):

    cov = np.random.normal(0, 1, (N, 1))
    W = np.concatenate((W, cov), axis =1)

W           = tf.convert_to_tensor( W,                      dtype = tf.float32 )
beta_0      = tf.convert_to_tensor( [[-np.log(N-1)], [+0]], dtype = tf.float32 )  
beta_lambda = tf.convert_to_tensor( [[-1],           [+2]], dtype = tf.float32 ) 
beta_gamma  = tf.convert_to_tensor( [[-1],           [-1]], dtype = tf.float32 ) 

initial_distribution  = AB_initial(W, beta_0)
K_eta = AB_SIS_transition(W, beta_lambda, beta_gamma)
q     = tf.convert_to_tensor([[0.8], [0.8]], dtype = tf.float32)

Exp = compartmental_model(N, M, initial_distribution, K_eta, q)
C, Y = Exp.run(T)
while (tf.reduce_sum( tf.reduce_sum(C, axis =0), axis =1)[0]>(N*(T+1)-100)):
    C, Y = Exp.run(T)

i = -1
for q_value in q_list:

    i = i +1

    q     = tf.convert_to_tensor([[q_value], [q_value]], dtype = tf.float32)
    ########################################
    # Whiteley-Rimella approximation
    pi_0 = mean_SIS_initial(initial_distribution)
    K_eta_mean = mean_SIS_transition(K_eta)

    multi_approx = multinomial_approximation(N, M, pi_0, K_eta_mean, q)
    pi, pi_T = multi_approx.run(Y)

    K_eta_multi = AB_SIS_transition_multi(W, beta_lambda, beta_gamma)

    for k in range(replicates):
        APF = proposal_APF(Nx, N, M, initial_distribution, K_eta_multi, q,)
        SMC_APF = smc(APF)

        _, ESS_history, loglikelihood_APF = SMC_APF.run(Y)

        j = 0
        loglikelihood_NotDGP[i, j, k] = tf.reduce_sum(loglikelihood_APF).numpy()
        ESS_NotDGP[i, j, k, :]        = ESS_history

        for h in h_list:
            j = j +1

            proposal = alternative_proposal(Nx, N, M, initial_distribution, K_eta, K_eta_multi, q, h, pi_T)
            SMC_proposal = smc(proposal)

            startTime = time.time()
            _, ESS_history, loglikelihood = SMC_proposal.run(Y)
            executionTime = (time.time() - startTime)            

            loglikelihood_NotDGP[i, j, k] = (tf.reduce_sum(loglikelihood).numpy())
            ESS_NotDGP[i, j, k, :]        = ESS_history


np.save("AlternativeProposal/Data/Output/LikelihoodEmission/LikelihoodEmission_SIS_DGP_numpy",    loglikelihood_DGP)
np.save("AlternativeProposal/Data/Output/LikelihoodEmission/ESSEmission_SIS_DGP_numpy",          ESS_DGP)
np.save("AlternativeProposal/Data/Output/LikelihoodEmission/LikelihoodEmission_SIS_NotDGP_numpy", loglikelihood_NotDGP)
np.save("AlternativeProposal/Data/Output/LikelihoodEmission/ESSEmission_SIS_NotDGP_numpy",       ESS_NotDGP)
