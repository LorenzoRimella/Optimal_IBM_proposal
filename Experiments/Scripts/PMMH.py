import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from synthetic_data import *
from proposal_epidemic import *
from smc import *


def alternative_proposal_SIS(Nx, N, M, W, h, Y):

    def proposal_without_parameter(parameter_sample):
        beta_0      = tf.reshape(parameter_sample[0], (2, 1))  
        beta_lambda = tf.reshape(parameter_sample[1], (2, 1)) 
        beta_gamma  = tf.reshape(parameter_sample[2], (2, 1)) 
        q           = tf.reshape(parameter_sample[3], (2, 1)) 

        initial_distribution  = AB_initial(W, beta_0)
        K_eta                 = AB_SIS_transition(W, beta_lambda, beta_gamma)

        pi_0         = mean_SIS_initial(initial_distribution)
        K_eta_mean   = mean_SIS_transition(K_eta) 
        multi_approx = multinomial_approximation(N, M, pi_0, K_eta_mean, q)
        _, pi_T      = multi_approx.run(Y)

        K_eta_multi = AB_SIS_transition_multi(W, beta_lambda, beta_gamma)
        proposal    = alternative_proposal(Nx, N, M, initial_distribution, K_eta, K_eta_multi, q, h, pi_T)

        return proposal

    return proposal_without_parameter

def alternative_proposal_SIS_fix0(Nx, N, M, W, h, Y):

    def proposal_without_parameter(parameter_sample):
        beta_0      = tf.convert_to_tensor( [[-np.log(N-1)], [+0]], dtype = tf.float32 ) 
        beta_lambda = tf.reshape(parameter_sample[0], (2, 1)) 
        beta_gamma  = tf.reshape(parameter_sample[1], (2, 1)) 
        q           = tf.reshape(parameter_sample[2], (2, 1)) 

        initial_distribution  = AB_initial(W, beta_0)
        K_eta                 = AB_SIS_transition(W, beta_lambda, beta_gamma)

        pi_0         = mean_SIS_initial(initial_distribution)
        K_eta_mean   = mean_SIS_transition(K_eta) 
        multi_approx = multinomial_approximation(N, M, pi_0, K_eta_mean, q)
        _, pi_T      = multi_approx.run(Y)

        K_eta_multi = AB_SIS_transition_multi(W, beta_lambda, beta_gamma)
        proposal    = alternative_proposal(Nx, N, M, initial_distribution, K_eta, K_eta_multi, q, h, pi_T)

        return proposal

    return proposal_without_parameter


def alternative_proposal_SEIR(Nx, N, M, W, h, Y):
    
    def proposal_without_parameter(parameter_sample):
        beta_0      = tf.reshape(parameter_sample[0], (2, 1))  
        beta_lambda = tf.reshape(parameter_sample[1], (2, 1)) 
        rho         = tf.reshape(parameter_sample[2], (1, 1)) 
        beta_gamma  = tf.reshape(parameter_sample[3], (2, 1)) 
        q           = tf.reshape([0., 0., parameter_sample[4][0] , parameter_sample[4][1]], (4, 1)) 

        initial_distribution  = AB_SEIR_initial(W, beta_0)
        K_eta                 = AB_SEIR_transition(W, beta_lambda, rho, beta_gamma)

        pi_0         = mean_SEIR_initial(initial_distribution)
        K_eta_mean   = mean_SEIR_transition(K_eta) 
        multi_approx = multinomial_approximation(N, M, pi_0, K_eta_mean, q)
        _, pi_T      = multi_approx.run(Y)

        K_eta_multi = AB_SEIR_transition_multi(W, beta_lambda, rho, beta_gamma)
        proposal    = alternative_proposal(Nx, N, M, initial_distribution, K_eta, K_eta_multi, q, h, pi_T)

        return proposal

    return proposal_without_parameter


def alternative_proposal_SEIR_fixrho(Nx, N, M, W, h, Y):
    
    def proposal_without_parameter(parameter_sample):
        beta_0      = tf.reshape(parameter_sample[0], (2, 1))  
        beta_lambda = tf.reshape(parameter_sample[1], (2, 1)) 
        rho         = tf.reshape([0.2], (1, 1)) 
        beta_gamma  = tf.reshape(parameter_sample[2], (2, 1)) 
        q           = tf.reshape([0., 0., parameter_sample[3][0] , parameter_sample[3][1]], (4, 1)) 

        initial_distribution  = AB_SEIR_initial(W, beta_0)
        K_eta                 = AB_SEIR_transition(W, beta_lambda, rho, beta_gamma)

        pi_0         = mean_SEIR_initial(initial_distribution)
        K_eta_mean   = mean_SEIR_transition(K_eta) 
        multi_approx = multinomial_approximation(N, M, pi_0, K_eta_mean, q)
        _, pi_T      = multi_approx.run(Y)

        K_eta_multi = AB_SEIR_transition_multi(W, beta_lambda, rho, beta_gamma)
        proposal    = alternative_proposal(Nx, N, M, initial_distribution, K_eta, K_eta_multi, q, h, pi_T)

        return proposal

    return proposal_without_parameter

def alternative_proposal_SEIR_fix0(Nx, N, M, W, h, Y):
    
    def proposal_without_parameter(parameter_sample):
        beta_0      = tf.convert_to_tensor( [[-np.log(N-1)], [+0]], dtype = tf.float32 ) 
        beta_lambda = tf.reshape(parameter_sample[0], (2, 1)) 
        rho         = tf.reshape(parameter_sample[1], (1, 1)) 
        beta_gamma  = tf.reshape(parameter_sample[2], (2, 1)) 
        q           = tf.reshape([0., 0., parameter_sample[3][0] , parameter_sample[3][1]], (4, 1)) 

        initial_distribution  = AB_SEIR_initial(W, beta_0)
        K_eta                 = AB_SEIR_transition(W, beta_lambda, rho, beta_gamma)

        pi_0         = mean_SEIR_initial(initial_distribution)
        K_eta_mean   = mean_SEIR_transition(K_eta) 
        multi_approx = multinomial_approximation(N, M, pi_0, K_eta_mean, q)
        _, pi_T      = multi_approx.run(Y)

        K_eta_multi = AB_SEIR_transition_multi(W, beta_lambda, rho, beta_gamma)
        proposal    = alternative_proposal(Nx, N, M, initial_distribution, K_eta, K_eta_multi, q, h, pi_T)

        return proposal

    return proposal_without_parameter

def APF_SIS(Nx, N, M, W):
    
    def proposal_without_parameter(parameter_sample):
        beta_0      = tf.reshape(parameter_sample[0], (2, 1))  
        beta_lambda = tf.reshape(parameter_sample[1], (2, 1)) 
        beta_gamma  = tf.reshape(parameter_sample[2], (2, 1)) 
        q           = tf.reshape(parameter_sample[3], (2, 1)) 

        initial_distribution  = AB_initial(W, beta_0)

        K_eta_multi = AB_SIS_transition_multi(W, beta_lambda, beta_gamma)
        proposal    = proposal_APF(Nx, N, M, initial_distribution, K_eta_multi, q,)

        return proposal

    return proposal_without_parameter


class likelihood_parameter():

    def __init__(self, proposal_without_parameter):
    
        self.proposal_without_parameter = proposal_without_parameter

    def loglikelihood(self, parameter_sample, Y):

        proposal = self.proposal_without_parameter(parameter_sample)

        SMC_proposal = smc(proposal)
        _, _, loglikelihood = SMC_proposal.run(Y)

        return tf.reduce_sum(loglikelihood, keepdims=True)


class prior_parameter():
    
    def __init__(self, prior_list):

        self.prior_list = prior_list

    def sample(self):

        prior_sample = []
        for prior_dist in self.prior_list:

            prior_sample.append(prior_dist.sample())
        
        return prior_sample

    def logpdf(self, parameter_sample):

        log_likelihood = tf.convert_to_tensor([0], dtype = tf.float32)
        for i in range(len(self.prior_list)):
            log_likelihood = log_likelihood + tf.reduce_sum(tf.math.log(self.prior_list[i].prob(parameter_sample[i])))

        return log_likelihood


def Gaussian_RW(scale_star):
    
    def proposal(loc_star, scale_by):

        return tfp.distributions.Normal(loc = loc_star, scale = [scale_star[i]/scale_by for i in range(len(scale_star))])

    return proposal

def identity(a):

    return a

def logit(a):

    return tf.math.log(a/(1-a))
    
def invlogit(a):

    return tf.exp(a)/(1+tf.exp(a))

class proposal_parameter():

    def __init__(self, proposal_list, to_proposal_space_list, to_parameter_space_list):

        self.poposal_list            = proposal_list
        self.to_proposal_space_list  = to_proposal_space_list
        self.to_parameter_space_list = to_parameter_space_list

    def sample(self, parameter_sample, scale_by):

        proposal_sample = []
        for i in range(len(self.poposal_list)):

            parameter_sample_to_proposal = self.to_proposal_space_list[i](parameter_sample[i])

            proposal_sample_to_parameter = self.to_parameter_space_list[i](self.poposal_list[i](parameter_sample_to_proposal, scale_by).sample())

            proposal_sample.append(proposal_sample_to_parameter)
        
        return proposal_sample

    def logpdf(self, parameter_sample, proposal_sample, scale_by):
    
        log_likelihood_ratio = tf.convert_to_tensor([0], dtype = tf.float32)
        for i in range(len(self.poposal_list)):

            parameter_sample_to_proposal       = self.to_proposal_space_list[i](parameter_sample[i])
            proposal_sample_sample_to_proposal = self.to_proposal_space_list[i](proposal_sample[i])

            log_likelihood_ratio = log_likelihood_ratio + tf.reduce_sum(tf.math.log(self.poposal_list[i](proposal_sample_sample_to_proposal, scale_by).prob(parameter_sample_to_proposal)) - tf.math.log(self.poposal_list[i](parameter_sample_to_proposal, scale_by).prob(proposal_sample_sample_to_proposal)))

        return log_likelihood_ratio


class PMMH:

    def __init__(self, prior_parameter, proposal_parameter, likelihood_parameter):

        self.prior_parameter      = prior_parameter
        self.proposal_parameter   = proposal_parameter
        self.likelihood_parameter = likelihood_parameter

    def step_0(self, Y):
    
        parameter_sample               = self.prior_parameter.sample()
        parameter_sample_logprior      = self.prior_parameter.logpdf(parameter_sample)
        parameter_sample_loglikelihood = self.likelihood_parameter.loglikelihood(parameter_sample, Y)

        while tf.math.is_nan(parameter_sample_loglikelihood):
            parameter_sample               = self.prior_parameter.sample()
            parameter_sample_logprior      = self.prior_parameter.logpdf(parameter_sample)
            parameter_sample_loglikelihood = self.likelihood_parameter.loglikelihood(parameter_sample, Y)

        return parameter_sample, parameter_sample_logprior, parameter_sample_loglikelihood

    def step_i(self, parameter_sample, parameter_sample_logprior, parameter_sample_loglikelihood, Y, scale_by):

        proposal_sample               = self.proposal_parameter.sample(parameter_sample, scale_by)
        proposal_sample_logprior      = self.prior_parameter.logpdf(proposal_sample)
        proposal_sample_loglikelihood = self.likelihood_parameter.loglikelihood(proposal_sample, Y)

        while tf.math.is_nan(proposal_sample_loglikelihood):
            proposal_sample               = self.proposal_parameter.sample(parameter_sample, scale_by)
            proposal_sample_logprior      = self.prior_parameter.logpdf(proposal_sample)
            proposal_sample_loglikelihood = self.likelihood_parameter.loglikelihood(proposal_sample, Y)

        prior_logratio       = proposal_sample_logprior - parameter_sample_logprior
        proposal_logratio    = self.proposal_parameter.logpdf(parameter_sample, proposal_sample, scale_by)
        likelihood_logration = proposal_sample_loglikelihood - parameter_sample_loglikelihood

        log_acceptance = (prior_logratio + proposal_logratio + likelihood_logration) #*(1-tf.cast(tf.math.is_nan(proposal_sample_loglikelihood), dtype = tf.float32))

        logU           = tf.math.log(tfp.distributions.Uniform(low = 0., high = 1.).sample())
        
        parameter_sample               = [(tf.cast((logU>log_acceptance), dtype = tf.float32)*parameter_sample[i]               + tf.cast((logU<log_acceptance), dtype = tf.float32)*proposal_sample[i]) for i in range(len(parameter_sample))]
        parameter_sample_logprior      =   tf.cast((logU>log_acceptance), dtype = tf.float32)*parameter_sample_logprior         + tf.cast((logU<log_acceptance), dtype = tf.float32)*proposal_sample_logprior
        parameter_sample_loglikelihood =   tf.cast((logU>log_acceptance), dtype = tf.float32)*parameter_sample_loglikelihood    + tf.cast((logU<log_acceptance), dtype = tf.float32)*proposal_sample_loglikelihood

        return parameter_sample, parameter_sample_logprior, parameter_sample_loglikelihood, tf.cast((logU<log_acceptance), dtype = tf.float32)

    def run(self, Y, iterations, checkfilename, outputfile_name, save_every, scale_by_rate = 2):
    
        parameter_sample, parameter_sample_logprior, parameter_sample_loglikelihood = self.step_0(Y)

        parameter_chain = tf.reshape(tf.concat(parameter_sample, axis =0), (tf.concat(parameter_sample, axis =0).shape[0], 1))

        nr_acceepted = 0
        scale_by = 1
        for i in range(1, iterations):

            parameter_sample, parameter_sample_logprior, parameter_sample_loglikelihood, accepted = self.step_i(parameter_sample, parameter_sample_logprior, parameter_sample_loglikelihood, Y, scale_by)

            nr_acceepted = nr_acceepted + accepted
            if i%save_every == 0:
                parameter_sample_loglikelihood = self.likelihood_parameter.loglikelihood(parameter_sample, Y)
                string = ["Acceptance rate "+ str(nr_acceepted.numpy()/save_every), "\n"]
                f= open(checkfilename, "a")
                f.writelines(string)
                f.close()

                nr_acceepted = 0

                # if ((nr_acceepted.numpy()/save_every)<0.17): # if (((nr_acceepted.numpy()/i)<0.15) and (i%(5*save_every))) == 0:
                #     nr_acceepted = 0
                #     scale_by = scale_by*scale_by_rate
                #     string = ["Scale by "+ str(scale_by), "\n"]
                #     f= open(checkfilename, "a")
                #     f.writelines(string)
                #     f.close()

                # elif ((nr_acceepted.numpy()/save_every)>0.29): # elif (((nr_acceepted.numpy()/i)>0.4) and (i%(5*save_every))) == 0:
                #     nr_acceepted = 0
                #     scale_by = scale_by/scale_by_rate
                #     string = ["Scale by "+ str(scale_by), "\n"]
                #     f= open(checkfilename, "a")
                #     f.writelines(string)
                #     f.close()
                
                # else:
                #     nr_acceepted = 0

                np.save(outputfile_name, parameter_chain)

            parameter_chain = tf.concat((parameter_chain, tf.reshape(tf.concat(parameter_sample, axis =0), (tf.concat(parameter_sample, axis =0).shape[0], 1))), axis = 1)

        return parameter_chain

    def run_from(self, Y, parameter_sample, iterations, checkfilename, outputfile_name, save_every, scale_by_rate = 2):

        parameter_sample_logprior      = self.prior_parameter.logpdf(parameter_sample)
        parameter_sample_loglikelihood = self.likelihood_parameter.loglikelihood(parameter_sample, Y)

        while tf.math.is_nan(parameter_sample_loglikelihood):
            parameter_sample_logprior      = self.prior_parameter.logpdf(parameter_sample)
            parameter_sample_loglikelihood = self.likelihood_parameter.loglikelihood(parameter_sample, Y)

        parameter_chain = tf.reshape(tf.concat(parameter_sample, axis =0), (tf.concat(parameter_sample, axis =0).shape[0], 1))

        nr_acceepted = 0
        scale_by = 1
        for i in range(1, iterations):

            parameter_sample, parameter_sample_logprior, parameter_sample_loglikelihood, accepted = self.step_i(parameter_sample, parameter_sample_logprior, parameter_sample_loglikelihood, Y, scale_by)

            nr_acceepted = nr_acceepted + accepted
            if i%save_every == 0:
                parameter_sample_loglikelihood = self.likelihood_parameter.loglikelihood(parameter_sample, Y)
                string = ["Acceptance rate "+ str(nr_acceepted.numpy()/save_every), "\n"]
                f= open(checkfilename, "a")
                f.writelines(string)
                f.close()

                nr_acceepted = 0

                # if ((nr_acceepted.numpy()/save_every)<0.17): # if (((nr_acceepted.numpy()/i)<0.15) and (i%(5*save_every))) == 0:
                #     nr_acceepted = 0
                #     scale_by = scale_by*scale_by_rate
                #     string = ["Scale by "+ str(scale_by), "\n"]
                #     f= open(checkfilename, "a")
                #     f.writelines(string)
                #     f.close()

                # elif ((nr_acceepted.numpy()/save_every)>0.29): # elif (((nr_acceepted.numpy()/i)>0.4) and (i%(5*save_every))) == 0:
                #     nr_acceepted = 0
                #     scale_by = scale_by/scale_by_rate
                #     string = ["Scale by "+ str(scale_by), "\n"]
                #     f= open(checkfilename, "a")
                #     f.writelines(string)
                #     f.close()
                
                # else:
                #     nr_acceepted = 0

                np.save(outputfile_name, parameter_chain)

            parameter_chain = tf.concat((parameter_chain, tf.reshape(tf.concat(parameter_sample, axis =0), (tf.concat(parameter_sample, axis =0).shape[0], 1))), axis = 1)

        return parameter_chain



class proposal_parameter_MwG():
    
    def __init__(self, proposal_list, to_proposal_space_list, to_parameter_space_list):

        self.poposal_list            = proposal_list
        self.to_proposal_space_list  = to_proposal_space_list
        self.to_parameter_space_list = to_parameter_space_list

    def sample(self, index, parameter_sample, scale_by):

        proposal_sample = []
        for i in range(len(self.poposal_list)):

            parameter_sample_to_proposal = self.to_proposal_space_list[i](parameter_sample[i])

            proposal_sample_to_parameter = parameter_sample[i]*tf.cast(i!=index, dtype = tf.float32) + self.to_parameter_space_list[i](self.poposal_list[i](parameter_sample_to_proposal, scale_by).sample())*tf.cast(i==index, dtype = tf.float32) 

            proposal_sample.append(proposal_sample_to_parameter)
        
        return proposal_sample

    def logpdf(self, parameter_sample, proposal_sample, scale_by):
    
        log_likelihood_ratio = tf.convert_to_tensor([0], dtype = tf.float32)
        for i in range(len(self.poposal_list)):

            parameter_sample_to_proposal       = self.to_proposal_space_list[i](parameter_sample[i])
            proposal_sample_sample_to_proposal = self.to_proposal_space_list[i](proposal_sample[i])

            log_likelihood_ratio = log_likelihood_ratio + tf.reduce_sum(tf.math.log(self.poposal_list[i](proposal_sample_sample_to_proposal, scale_by).prob(parameter_sample_to_proposal)) - tf.math.log(self.poposal_list[i](parameter_sample_to_proposal, scale_by).prob(proposal_sample_sample_to_proposal)))

        return log_likelihood_ratio



class PMMwG:
    
    def __init__(self, prior_parameter, proposal_parameter, likelihood_parameter):

        self.prior_parameter      = prior_parameter
        self.proposal_parameter   = proposal_parameter
        self.likelihood_parameter = likelihood_parameter

    def step_0(self, Y):
    
        parameter_sample               = self.prior_parameter.sample()
        parameter_sample_logprior      = self.prior_parameter.logpdf(parameter_sample)
        parameter_sample_loglikelihood = self.likelihood_parameter.loglikelihood(parameter_sample, Y)

        while tf.math.is_nan(parameter_sample_loglikelihood):
            parameter_sample               = self.prior_parameter.sample()
            parameter_sample_logprior      = self.prior_parameter.logpdf(parameter_sample)
            parameter_sample_loglikelihood = self.likelihood_parameter.loglikelihood(parameter_sample, Y)

        return parameter_sample, parameter_sample_logprior, parameter_sample_loglikelihood

    def step_i(self, index, parameter_sample, parameter_sample_logprior, parameter_sample_loglikelihood, Y, scale_by):

        proposal_sample               = self.proposal_parameter.sample(index, parameter_sample, scale_by)
        proposal_sample_logprior      = self.prior_parameter.logpdf(proposal_sample)
        proposal_sample_loglikelihood = self.likelihood_parameter.loglikelihood(proposal_sample, Y)

        while tf.math.is_nan(proposal_sample_loglikelihood):
            proposal_sample               = self.proposal_parameter.sample(index, parameter_sample, scale_by)
            proposal_sample_logprior      = self.prior_parameter.logpdf(proposal_sample)
            proposal_sample_loglikelihood = self.likelihood_parameter.loglikelihood(proposal_sample, Y)

        prior_logratio       = proposal_sample_logprior - parameter_sample_logprior
        proposal_logratio    = self.proposal_parameter.logpdf(parameter_sample, proposal_sample, scale_by)
        likelihood_logration = proposal_sample_loglikelihood - parameter_sample_loglikelihood

        log_acceptance = (prior_logratio + proposal_logratio + likelihood_logration) #*(1-tf.cast(tf.math.is_nan(proposal_sample_loglikelihood), dtype = tf.float32))

        logU           = tf.math.log(tfp.distributions.Uniform(low = 0., high = 1.).sample())
        
        parameter_sample               = [(tf.cast((logU>log_acceptance), dtype = tf.float32)*parameter_sample[i]               + tf.cast((logU<log_acceptance), dtype = tf.float32)*proposal_sample[i]) for i in range(len(parameter_sample))]
        parameter_sample_logprior      =   tf.cast((logU>log_acceptance), dtype = tf.float32)*parameter_sample_logprior         + tf.cast((logU<log_acceptance), dtype = tf.float32)*proposal_sample_logprior
        parameter_sample_loglikelihood =   tf.cast((logU>log_acceptance), dtype = tf.float32)*parameter_sample_loglikelihood    + tf.cast((logU<log_acceptance), dtype = tf.float32)*proposal_sample_loglikelihood

        return parameter_sample, parameter_sample_logprior, parameter_sample_loglikelihood, tf.cast((logU<log_acceptance), dtype = tf.float32)

    def run(self, Y, iterations, checkfilename, outputfile_name, save_every, scale_by_rate = 2):
    
        parameter_sample, parameter_sample_logprior, parameter_sample_loglikelihood = self.step_0(Y)

        parameter_chain = tf.reshape(tf.concat(parameter_sample, axis =0), (tf.concat(parameter_sample, axis =0).shape[0], 1))

        nr_acceepted = 0
        scale_by = 1
        i = 0
        while i<iterations:

            for index in range(len(parameter_sample)):
                i = i +1
                parameter_sample, parameter_sample_logprior, parameter_sample_loglikelihood, accepted = self.step_i(index, parameter_sample, parameter_sample_logprior, parameter_sample_loglikelihood, Y, scale_by)

                nr_acceepted = nr_acceepted + accepted
                if i%save_every == 0:
                    string = ["Acceptance rate "+ str(nr_acceepted.numpy()/save_every), "\n"]
                    f= open(checkfilename, "a")
                    f.writelines(string)
                    f.close()

                    nr_acceepted = 0

                    # if ((nr_acceepted.numpy()/save_every)<0.17): # if (((nr_acceepted.numpy()/i)<0.15) and (i%(5*save_every))) == 0:
                    #     nr_acceepted = 0
                    #     scale_by = scale_by*scale_by_rate
                    #     string = ["Scale by "+ str(scale_by), "\n"]
                    #     f= open(checkfilename, "a")
                    #     f.writelines(string)
                    #     f.close()

                    # elif ((nr_acceepted.numpy()/save_every)>0.29): # elif (((nr_acceepted.numpy()/i)>0.4) and (i%(5*save_every))) == 0:
                    #     nr_acceepted = 0
                    #     scale_by = scale_by/scale_by_rate
                    #     string = ["Scale by "+ str(scale_by), "\n"]
                    #     f= open(checkfilename, "a")
                    #     f.writelines(string)
                    #     f.close()
                    
                    # else:
                    #     nr_acceepted = 0

                    np.save(outputfile_name, parameter_chain)

                parameter_chain = tf.concat((parameter_chain, tf.reshape(tf.concat(parameter_sample, axis =0), (tf.concat(parameter_sample, axis =0).shape[0], 1))), axis = 1)

        return parameter_chain

    def run_from(self, Y, parameter_sample, iterations, checkfilename, outputfile_name, save_every, scale_by_rate = 2):

        parameter_sample_logprior      = self.prior_parameter.logpdf(parameter_sample)
        parameter_sample_loglikelihood = self.likelihood_parameter.loglikelihood(parameter_sample, Y)

        while tf.math.is_nan(parameter_sample_loglikelihood):
            parameter_sample_logprior      = self.prior_parameter.logpdf(parameter_sample)
            parameter_sample_loglikelihood = self.likelihood_parameter.loglikelihood(parameter_sample, Y)

        parameter_chain = tf.reshape(tf.concat(parameter_sample, axis =0), (tf.concat(parameter_sample, axis =0).shape[0], 1))

        nr_acceepted = 0
        scale_by = 1
        i = 0
        while i<iterations:

            for index in range(len(parameter_sample)):
                i = i +1
                parameter_sample, parameter_sample_logprior, parameter_sample_loglikelihood, accepted = self.step_i(index, parameter_sample, parameter_sample_logprior, parameter_sample_loglikelihood, Y, scale_by)

                nr_acceepted = nr_acceepted + accepted
                if i%save_every == 0:
                    string = ["Acceptance rate "+ str(nr_acceepted.numpy()/save_every), "\n"]
                    f= open(checkfilename, "a")
                    f.writelines(string)
                    f.close()

                    nr_acceepted = 0

                    # if ((nr_acceepted.numpy()/save_every)<0.17): # if (((nr_acceepted.numpy()/i)<0.15) and (i%(5*save_every))) == 0:
                    #     nr_acceepted = 0
                    #     scale_by = scale_by*scale_by_rate
                    #     string = ["Scale by "+ str(scale_by), "\n"]
                    #     f= open(checkfilename, "a")
                    #     f.writelines(string)
                    #     f.close()

                    # elif ((nr_acceepted.numpy()/save_every)>0.29): # elif (((nr_acceepted.numpy()/i)>0.4) and (i%(5*save_every))) == 0:
                    #     nr_acceepted = 0
                    #     scale_by = scale_by/scale_by_rate
                    #     string = ["Scale by "+ str(scale_by), "\n"]
                    #     f= open(checkfilename, "a")
                    #     f.writelines(string)
                    #     f.close()
                    
                    # else:
                    #     nr_acceepted = 0

                    np.save(outputfile_name, parameter_chain)

                parameter_chain = tf.concat((parameter_chain, tf.reshape(tf.concat(parameter_sample, axis =0), (tf.concat(parameter_sample, axis =0).shape[0], 1))), axis = 1)

        return parameter_chain
