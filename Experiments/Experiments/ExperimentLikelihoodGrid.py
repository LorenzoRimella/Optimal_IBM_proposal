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

T_list     = [10, 50, 100] 

Nx         = 512
h_list     = [5, 10]

beta_lambda_grid = np.linspace(-4, 4, 50)
beta_gamma_grid  = np.linspace(-4, 4, 50)

loglikelihood_lambda    = np.zeros((beta_lambda_grid.shape[0], beta_lambda_grid.shape[0], len(T_list), len(h_list)))
loglikelihood_gamma     = np.zeros((beta_gamma_grid.shape[0],  beta_gamma_grid.shape[0],  len(T_list), len(h_list)))


t_index = -1
for T in T_list:

    t_index = t_index + 1

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


    for i in range(loglikelihood_lambda.shape[0]):

        for j in range(loglikelihood_lambda.shape[1]):

            beta_lambda = tf.convert_to_tensor( [[beta_lambda_grid[i]], [beta_lambda_grid[j]]], dtype = tf.float32 ) 
            beta_gamma  = tf.convert_to_tensor( [[-1],                  [-1]], dtype = tf.float32 ) 

            K_eta = AB_SIS_transition(W, beta_lambda, beta_gamma)

            ########################################
            # Whiteley-Rimella approximation
            pi_0 = mean_SIS_initial(initial_distribution)
            K_eta_mean = mean_SIS_transition(K_eta)

            multi_approx = multinomial_approximation(N, M, pi_0, K_eta_mean, q)
            pi, pi_T = multi_approx.run(Y)

            K_eta_multi = AB_SIS_transition_multi(W, beta_lambda, beta_gamma)

            for h_index in range(len(h_list)):

                proposal = alternative_proposal(Nx, N, M, initial_distribution, K_eta, K_eta_multi, q, h_list[h_index], pi_T)
                SMC_proposal = smc(proposal)

                _, _, loglikelihood = SMC_proposal.run(Y)            

                loglikelihood_lambda[i, j, t_index, h_index] = (tf.reduce_sum(loglikelihood).numpy())

    np.save("AlternativeProposal/Data/Output/LikelihoodGrid/loglikelihood_lambda", loglikelihood_lambda)




t_index = -1
for T in T_list:

    t_index = t_index + 1

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


    for i in range(loglikelihood_lambda.shape[0]):

        for j in range(loglikelihood_lambda.shape[1]):

            beta_lambda = tf.convert_to_tensor( [[-1],                 [+2]], dtype = tf.float32 ) 
            beta_gamma  = tf.convert_to_tensor( [[beta_gamma_grid[i]], [beta_gamma_grid[j]]], dtype = tf.float32 ) 

            K_eta = AB_SIS_transition(W, beta_lambda, beta_gamma)

            ########################################
            # Whiteley-Rimella approximation
            pi_0 = mean_SIS_initial(initial_distribution)
            K_eta_mean = mean_SIS_transition(K_eta)

            multi_approx = multinomial_approximation(N, M, pi_0, K_eta_mean, q)
            pi, pi_T = multi_approx.run(Y)

            K_eta_multi = AB_SIS_transition_multi(W, beta_lambda, beta_gamma)

            for h_index in range(len(h_list)):

                proposal = alternative_proposal(Nx, N, M, initial_distribution, K_eta, K_eta_multi, q, h_list[h_index], pi_T)
                SMC_proposal = smc(proposal)

                _, _, loglikelihood = SMC_proposal.run(Y)            

                loglikelihood_gamma[i, j, t_index, h_index] = (tf.reduce_sum(loglikelihood).numpy())

    np.save("AlternativeProposal/Data/Output/LikelihoodGrid/loglikelihood_gamma", loglikelihood_gamma)


np.save("AlternativeProposal/Data/Output/LikelihoodGrid/loglikelihood_lambda", loglikelihood_lambda)
np.save("AlternativeProposal/Data/Output/LikelihoodGrid/loglikelihood_gamma", loglikelihood_gamma)