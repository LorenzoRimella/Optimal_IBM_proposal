import tensorflow as tf
import tensorflow_probability as tfp

def AB_initial(W, beta_0):

    N = W.shape[0]
    initial__n = 1/(1+tf.exp(-tf.einsum("nj,jk->nk", W, beta_0)))
    rate_0   = tf.reshape(initial__n, (N, 1))       
    
    initial_distr    = tf.concat((1-rate_0, rate_0), axis = 1)

    return initial_distr

def AB_SIS_transition(W, beta_lambda, beta_gamma):

    def K_eta(sumC_t, h = 1):

        N = W.shape[0]

        lambda__n = 1/(1+tf.exp(-tf.einsum("ij,jk->ik", W, beta_lambda)))
        rate_SI   = tf.reshape(h*lambda__n*sumC_t[1]/N, (N, 1, 1))

        gamma__n  = 1/(1+tf.exp(-tf.einsum("ij,jk->ik", W, beta_gamma)))
        rate_IS   = tf.reshape(h*gamma__n, (N, 1, 1))

        K_eta_h__n_r1 = tf.concat((1 - rate_SI, rate_SI), axis = 2)
        K_eta_h__n_r2 = tf.concat((rate_IS, 1 - rate_IS), axis = 2)
        K_eta_h__n    = tf.concat((K_eta_h__n_r1, K_eta_h__n_r2), axis = 1)

        return K_eta_h__n

    return K_eta

    
def AB_SEIR_initial(W, beta_0):

    N = W.shape[0]
    initial__n = 1/(1+tf.exp(-tf.einsum("nj,jk->nk", W, beta_0)))
    rate_0   = tf.reshape(initial__n, (N, 1))       
    
    initial_distr    = tf.concat((1-rate_0, tf.zeros(rate_0.shape), rate_0, tf.zeros(rate_0.shape)), axis = 1)

    return initial_distr


def AB_SEIR_transition(W, beta_lambda, rho, beta_gamma):

    def K_eta(sumx_t, h = 1):
        N = W.shape[0]

        lambda__n = 1/(1+tf.exp(-tf.einsum("ij,jk->ik", W, beta_lambda)))
        rate_SE   = tf.reshape(h*lambda__n*sumx_t[2]/N, (N, 1, 1))

        rate_EI   = h*rho*tf.ones((N, 1, 1))

        gamma__n = 1/(1+tf.exp(-tf.einsum("ij,jk->ik", W, beta_gamma)))
        rate_IR   = tf.reshape(h*gamma__n, (N, 1, 1))

        K_eta_h__n_r1 = tf.concat((1 - rate_SE, rate_SE, tf.zeros((N, 1, 2))), axis = 2)
        K_eta_h__n_r2 = tf.concat((tf.zeros((N, 1, 1)), tf.exp(-rate_EI), 1 - tf.exp(-rate_EI), tf.zeros((N, 1, 1))), axis = 2)
        K_eta_h__n_r3 = tf.concat((tf.zeros((N, 1, 2)), 1 - rate_IR, rate_IR), axis = 2)
        K_eta_h__n_r4 = tf.concat((tf.zeros((N, 1, 3)), tf.ones((N, 1, 1))), axis = 2)
        K_eta_h__n    = tf.concat((K_eta_h__n_r1, K_eta_h__n_r2, K_eta_h__n_r3, K_eta_h__n_r4), axis = 1)

        return K_eta_h__n

    return K_eta

    
class compartmental_model():

    def __init__(self, N, M, initial_distribution, K_eta, q):

        self.N     = N
        self.M     = M
        self.initial_distribution  = initial_distribution
        self.K_eta = K_eta
        self.q     = q

    def sim_C_0(self):

        cat = tfp.distributions.OneHotCategorical( probs = self.initial_distribution, dtype=tf.float32)

        return cat.sample() 


    def sim_C_t(self, C_tm1):

        sumC_tm1 = tf.reduce_sum( C_tm1, axis = 0 )

        prob_t = tf.einsum("ni,nik->nk", C_tm1, self.K_eta(sumC_tm1))

        cat = tfp.distributions.OneHotCategorical( probs = prob_t, dtype=tf.float32)

        return cat.sample()

    def sim_Y_t(self, C_t):
        
        prob_be = tf.einsum("ni,ik->nk", C_t, self.q)
        be = tfp.distributions.Bernoulli( probs = prob_be, dtype = tf.float32)

        return C_t*be.sample()

    def run(self, T):

        C_tm1 = self.sim_C_0()

        C = tf.reshape(C_tm1, (self.N, self.M, 1))
        Y = tf.zeros((self.N, self.M, 1))-10

        for t in range(0, T):

            C_tm1 = self.sim_C_t(C_tm1)
            C     = tf.concat((C, tf.reshape(C_tm1, (self.N, self.M, 1))), axis = 2)

            Y_t   = self.sim_Y_t(C_tm1)
            Y     = tf.concat((Y, tf.reshape(Y_t, (self.N, self.M, 1))), axis = 2)

        return C, Y
