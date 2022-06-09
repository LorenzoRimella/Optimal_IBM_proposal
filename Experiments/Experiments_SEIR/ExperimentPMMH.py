import sys
sys.path.append('AlternativeProposal/Scripts/')

from synthetic_data import *
from proposal_epidemic import *
from smc import *
from PMMH import *

import numpy as np


N = 1000
M = 4
covariates_n = 2

W = np.load("AlternativeProposal/Data/Input/W_SEIR.npy")
C = np.load("AlternativeProposal/Data/Input/C_SEIR.npy")
Y = np.load("AlternativeProposal/Data/Input/Y_SEIR.npy")

T     = 100

##############################################################################################
# Define the PMMH

# prior list
beta_0_prior      = tfp.distributions.Normal( loc  = [0., 0.], scale = [3., 3.])
beta_lambda_prior = tfp.distributions.Normal( loc  = [0., 0.], scale = [3., 3.])
rho_prior         = tfp.distributions.Gamma(  concentration  = [0.5], rate = [1.])
beta_gamma_prior  = tfp.distributions.Normal(      loc  = [0., 0.],        scale = [3., 3.])
q_prior           = tfp.distributions.Beta(     concentration1  = [5., 5.],         concentration0  = [5., 5.])

prior_list = [beta_0_prior, beta_lambda_prior, rho_prior, beta_gamma_prior, q_prior] # [beta_0_prior, beta_lambda_prior, beta_gamma_prior, q_prior] # [beta_lambda_prior, rho_prior, beta_gamma_prior, q_prior] # 

# proposal list
proposal_std = 0.025
beta_0_proposal      = Gaussian_RW([proposal_std, proposal_std])
beta_lambda_proposal = Gaussian_RW([proposal_std, proposal_std])
rho_proposal         = Gaussian_RW([proposal_std])
beta_gamma_proposal  = Gaussian_RW([proposal_std, proposal_std])
logit_q_proposal     = Gaussian_RW([proposal_std, proposal_std])

proposal_list = [beta_0_proposal, beta_lambda_proposal, rho_proposal, beta_gamma_proposal, logit_q_proposal] # [beta_0_proposal, beta_lambda_proposal, beta_gamma_proposal, logit_q_proposal] # [beta_lambda_proposal, rho_proposal, beta_gamma_proposal, logit_q_proposal] # 

# transform from parameters space to proposal space
beta_0_to_beta_0           = identity
beta_lambda_to_beta_lambda = identity
rho_to_logrho              = tf.math.log
beta_gamma_to_beta_gamma   = identity
q_to_logitq                = logit

to_proposal_space_list = [beta_0_to_beta_0, beta_lambda_to_beta_lambda, rho_to_logrho, beta_gamma_to_beta_gamma, q_to_logitq] # [beta_0_to_beta_0, beta_lambda_to_beta_lambda, beta_gamma_to_beta_gamma, q_to_logitq] # [beta_lambda_to_beta_lambda, rho_to_logrho, beta_gamma_to_beta_gamma, q_to_logitq] # 

# transform from proposal space to parameter space
beta_0_to_beta_0           = identity
beta_lambda_to_beta_lambda = identity
logrho_to_rho              = tf.math.exp
beta_gamma_to_beta_gamma   = identity
logitq_to_q                = invlogit

to_parameter_space_list = [beta_0_to_beta_0, beta_lambda_to_beta_lambda, logrho_to_rho, beta_gamma_to_beta_gamma, logitq_to_q] # [beta_0_to_beta_0, beta_lambda_to_beta_lambda, beta_gamma_to_beta_gamma, logitq_to_q] # [beta_lambda_to_beta_lambda, logrho_to_rho, beta_gamma_to_beta_gamma, logitq_to_q] # 

# define the proposal function
Nx = 2048
h  = 30

# APF or proposal fancy?
proposal_without_parameter = alternative_proposal_SEIR(Nx, N, M, W, h, Y) 

# define the PMMH components
prior_parameter_PMMH      = prior_parameter( prior_list)
proposal_parameter_PMMH   = proposal_parameter( proposal_list, to_proposal_space_list, to_parameter_space_list)
likelihood_parameter_PMMH = likelihood_parameter(proposal_without_parameter)

#############################################################
# Run the PMMH
PMMH_SEIR = PMMH(prior_parameter_PMMH, proposal_parameter_PMMH, likelihood_parameter_PMMH)

iterations      = 100000

# APF or proposal fancy
file_name       = "SEIR_PMMH_"+str(iterations)+"iter_fromLONG_"+str(h)+"h"
# file_name       = "PMMH_"+str(iterations)+"iter_"+"APF"

checkfile_name  = "AlternativeProposal/Check/PMMH/"+file_name+".txt"
outputfile_name = "AlternativeProposal/Data/Output/PMMH/"+file_name

save_every = 25

chain      = PMMH_SEIR.run(Y, iterations, checkfile_name, outputfile_name, save_every)

np.save(outputfile_name, chain)
