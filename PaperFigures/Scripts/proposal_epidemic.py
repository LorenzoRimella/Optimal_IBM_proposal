from pyrsistent import v
import tensorflow as tf
import tensorflow_probability as tfp

def AB_initial_multi(Nx, W, beta_0):

    N = W.shape[0]

    initial__n = 1/(1+tf.exp(-tf.einsum("ij,jk->ik", W, beta_0)))
    rate_0   = tf.reshape(tf.transpose(initial__n)*tf.ones((Nx, 1)), (Nx, N, 1))       
    
    initial_distr    = tf.concat((1-rate_0, rate_0), axis = 2)

    return initial_distr

def AB_SIS_transition_multi(W, beta_lambda, beta_gamma):

    def K_eta(sumx_t, h = 1):
        N = W.shape[0]

        lambda__n = 1/(1+tf.exp(-tf.einsum("ij,jk->ik", W, beta_lambda)))
        rate_SI   = tf.reshape(h*tf.transpose(lambda__n)*tf.reshape(sumx_t[:, 1]/N, (sumx_t.shape[0], 1)), (sumx_t.shape[0], N, 1, 1))

        gamma__n  = 1/(1+tf.exp(-tf.einsum("ij,jk->ik", W, beta_gamma)))
        rate_IS   = tf.reshape(h*tf.transpose(gamma__n)*tf.ones((sumx_t.shape[0], 1)), (sumx_t.shape[0], N, 1, 1))

        K_eta_h__n_r1 = tf.concat((1 - rate_SI, rate_SI), axis = 3)
        K_eta_h__n_r2 = tf.concat((rate_IS, 1 - rate_IS), axis = 3)
        K_eta_h__n    = tf.concat((K_eta_h__n_r1, K_eta_h__n_r2), axis = 2)

        return K_eta_h__n

    return K_eta

def mean_SIS_initial(initial_distr):
    
    return tf.transpose(tf.reduce_mean(initial_distr, axis = 0, keepdims = True))

def mean_SIS_transition(AB_K_eta):
    
    def K_eta(sumx_t, h = 1):

        return tf.reduce_mean(AB_K_eta(sumx_t, h), axis = 0)

    return K_eta

def AB_SEIR_transition_multi(W, beta_lambda, rho, beta_gamma):

    def K_eta(sumx_t, h = 1):
        N = W.shape[0]

        lambda__n = 1/(1+tf.exp(-tf.einsum("ij,jk->ik", W, beta_lambda)))
        rate_SE   = tf.reshape(h*tf.transpose(lambda__n)*tf.reshape(sumx_t[:, 2]/N, (sumx_t.shape[0], 1)), (sumx_t.shape[0], N, 1, 1))

        rate_EI   = h*rho*tf.ones((sumx_t.shape[0], N, 1, 1))

        gamma__n  = 1/(1+tf.exp(-tf.einsum("ij,jk->ik", W, beta_gamma)))
        rate_IR   = tf.reshape(h*tf.transpose(gamma__n)*tf.ones((sumx_t.shape[0], 1)), (sumx_t.shape[0], N, 1, 1))


        K_eta_h__n_r1 = tf.concat((1 - rate_SE, rate_SE, tf.zeros((sumx_t.shape[0], N, 1, 2))), axis = 3)
        K_eta_h__n_r2 = tf.concat((tf.zeros((sumx_t.shape[0], N, 1, 1)), tf.exp(-rate_EI), 1 - tf.exp(-rate_EI), tf.zeros((sumx_t.shape[0], N, 1, 1))), axis = 3)
        K_eta_h__n_r3 = tf.concat((tf.zeros((sumx_t.shape[0], N, 1, 2)), 1 - rate_IR, rate_IR), axis = 3)
        K_eta_h__n_r4 = tf.concat((tf.zeros((sumx_t.shape[0], N, 1, 3)), tf.ones((sumx_t.shape[0], N, 1, 1))), axis = 3)
        K_eta_h__n    = tf.concat((K_eta_h__n_r1, K_eta_h__n_r2, K_eta_h__n_r3, K_eta_h__n_r4), axis = 2)

        return K_eta_h__n

    return K_eta

def mean_SEIR_initial(initial_distr):
    
    return tf.transpose(tf.reduce_mean(initial_distr, axis = 0, keepdims = True))

def mean_SEIR_transition(AB_K_eta):
    
    def K_eta(sumx_t, h = 1):

        return tf.reduce_mean(AB_K_eta(sumx_t, h), axis = 0)

    return K_eta

def homogeneous_SEIR_transition( N, beta, rho, gamma):
    
    def K_eta(pi_t, h = 1):

        rate_SE   = tf.convert_to_tensor([[h*beta*pi_t[2,0]/N]])

        rate_EI   = tf.convert_to_tensor([[h*rho]])

        rate_IR   = tf.convert_to_tensor([[h*gamma]])

        K_eta_h__n_r1 = tf.concat((tf.exp(-rate_SE), 1 - tf.exp(-rate_SE), tf.zeros((1, 2))), axis = 1)
        K_eta_h__n_r2 = tf.concat((tf.zeros((1, 1)), tf.exp(-rate_EI), 1 - tf.exp(-rate_EI), tf.zeros((1, 1))), axis = 1)
        K_eta_h__n_r3 = tf.concat((tf.zeros((1, 2)), tf.exp(-rate_IR), 1 - tf.exp(-rate_IR)), axis = 1)
        K_eta_h__n_r4 = tf.concat((tf.zeros((1, 3)), tf.ones((1, 1))), axis = 1)
        K_eta_h__n    = tf.concat((K_eta_h__n_r1, K_eta_h__n_r2, K_eta_h__n_r3, K_eta_h__n_r4), axis = 0)

        return K_eta_h__n

    return K_eta

class multinomial_approximation():

    def __init__(self, N, M, pi_0, K_eta, q):

        self.N     = N
        self.M     = M
        self.pi_0  = tf.reshape(pi_0, (self.M, 1))
        self.K_eta = K_eta
        self.q     = q

    def step_t_tm1(self, pi_tm1):

        return tf.einsum("ji,jk->ki", pi_tm1, self.K_eta(self.N*pi_tm1))

    def step_t(self, pi_t_tm1, Y_t):

        Y_t_cumulative  = tf.transpose(tf.reduce_sum( Y_t, axis = 0, keepdims = True ))

        pi_t_normalized = pi_t_tm1*(1-self.q)/(1 - tf.reduce_sum(pi_t_tm1*self.q))

        return Y_t_cumulative/self.N + (1 - tf.reduce_sum(Y_t_cumulative/self.N))*pi_t_normalized

    def step_t_T(self, pi_t, pi_tp1_T):

        L_t = tf.transpose(pi_t*self.K_eta(self.N*pi_t)/ tf.einsum("ji,jk->ik", pi_t, self.K_eta(self.N*pi_t)))

        return tf.einsum("ji,jk->ki", pi_tp1_T, L_t)

    def run(self, Y):

        T  = Y.shape[2]

        pi_t_tm1 = self.pi_0

        pi_tm1   = pi_t_tm1

        pi     = pi_tm1

        for t in range(1, T):

            pi_t_tm1 = self.step_t_tm1(pi_tm1)

            Y_t      = Y[:,:,t]
            pi_tm1   = self.step_t(pi_t_tm1, Y_t)

            pi = tf.concat((pi, pi_tm1), axis =1)

        pi_tp1_T = pi_tm1
        pi_T     = pi_tp1_T

        for t in range(T-2, -1, -1):

            pi_tp1_T = self.step_t_T(tf.reshape(pi[:,t], (self.M, 1)), pi_tp1_T)
            pi_T = tf.concat((pi_tp1_T, pi_T), axis =1)

        return pi, pi_T

class alternative_proposal():
    
    def __init__(self, Nx, N, M, initial_distribution, K_eta_single, K_eta_multi, q, h, pi_T, Y = False, preproc = False):
        
        self.Nx    = Nx
        self.N     = N
        self.M     = M
        self.K_eta_single = K_eta_single
        self.K_eta_multi  = K_eta_multi
        if initial_distribution.shape[0] == self.M:
            self.initial_distribution  = initial_distribution*tf.ones((self.Nx, self.N, self.M))
        elif initial_distribution.shape[0] == self.N:
            self.initial_distribution  = tf.reshape(initial_distribution, (1, N, M))*tf.ones((Nx, N, M))
            
        self.q     = q

        self.h = h

        self.pi_T = pi_T 

        if preproc:
            self.Y = Y
            self.preprocessing()

    def run_tilde_e_s_tph(self, pi_s_tph, E_sp1_tph):

        return tf.einsum("nj,nij->ni", E_sp1_tph, self.K_eta_single(self.N*pi_s_tph))

    def run_e_s_tph(self, Y_s, tildeE_s_tph):
    
        return (tf.reshape(self.q, (1, self.M))*Y_s + (1-tf.reshape(self.q, (1, self.M)))*(1-tf.reduce_sum(Y_s, axis = 1, keepdims = True)))*tildeE_s_tph

    def run_e_t_tph_tildee_t_tph(self, t, Y_t_tph):
        
        pi_t_tph_T = self.pi_T[:, t:t+self.h+1]
        steps = Y_t_tph.shape[2]

        tildee_s_tph = tf.ones((self.N, self.M))
        e_s_tph      = self.run_e_s_tph(Y_t_tph[:, :, -1], tildee_s_tph)

        for s in range(steps-2, -1, -1):
            tildee_s_tph = self.run_tilde_e_s_tph(pi_t_tph_T[:, s], e_s_tph)
            e_s_tph      = self.run_e_s_tph(Y_t_tph[:, :, s], tildee_s_tph)


        return e_s_tph, tildee_s_tph

    def run_tildee_0_h(self, Y_t_tph):
        
        pi_t_tph_T = self.pi_T[:, 0:self.h+1]

        tildee_s_tph = tf.ones((self.N, self.M))
        e_s_tph      = tf.cast(self.h!=0, dtype = tf.float32)*self.run_e_s_tph(Y_t_tph[:, :, self.h], tildee_s_tph) + tf.cast(self.h==0, dtype = tf.float32)*tf.ones((self.N, self.M))

        for s in range(self.h-1, 0, -1):
            tildee_s_tph = self.run_tilde_e_s_tph(pi_t_tph_T[:, s], e_s_tph)
            e_s_tph      = self.run_e_s_tph(Y_t_tph[:, :, s], tildee_s_tph)

        s = 0
        tildee_s_tph = tf.cast(self.h!=0, dtype = tf.float32)*self.run_tilde_e_s_tph(pi_t_tph_T[:, s], e_s_tph) + tf.cast(self.h==0, dtype = tf.float32)*tf.ones((self.N, self.M))

        return tildee_s_tph

    def preprocessing(self):
        
        T = self.Y.shape[2]
        time_index = tf.cast(tf.linspace(tf.cast(1, dtype = tf.float32), tf.cast(T-1, dtype = tf.float32), tf.cast(T-1, dtype = tf.int32)), dtype = tf.int32)

        def run_e_t_tph_tildee_t_tph_givenY(t):

            return self.run_e_t_tph_tildee_t_tph(t, self.Y[   :, :, t:t+self.h+1])

        self.e_s_tph, self.tildee_s_tph = tf.map_fn(run_e_t_tph_tildee_t_tph_givenY, time_index, (tf.float32, tf.float32))

        self.tildee_0_h = self.run_tildee_0_h(self.Y[   :, :, 0:self.h+1])

    def step_0_preproc(self):
    
        normalizing = tf.einsum("ni, pni -> pn", self.tildee_0_h, self.initial_distribution)

        prob_0_1h = self.initial_distribution*tf.reshape(self.tildee_0_h, (1, self.N, self.M))/tf.reshape(normalizing, (self.Nx, self.N, 1) )

        x_0 = tfp.distributions.OneHotCategorical( probs = prob_0_1h, dtype=tf.float32).sample()

        logw_0        = tf.math.log(normalizing) - tf.math.log(tf.einsum("ni, pni -> pn", self.tildee_0_h, x_0))

        K_eta_particles_0 = self.K_eta_multi(tf.reduce_sum(x_0, axis = 1))
        normalizing_next = tf.einsum("pnj, nj -> pn", tf.einsum("pni, pnij -> pnj", x_0, K_eta_particles_0), self.e_s_tph[0, :, :])

        return x_0, tf.reduce_sum(logw_0, axis =1), tf.reduce_sum(tf.math.log(normalizing_next), axis =1) 

    def step_t_preproc(self, t, x_tm1):

        K_eta_particles_tm1 = self.K_eta_multi(tf.reduce_sum(x_tm1, axis = 1)) #tf.map_fn(self.K_eta, tf.reduce_sum(x_tm1, axis =1))
        normalizing     = tf.einsum("pnj, nj -> pn", tf.einsum("pni, pnij -> pnj", x_tm1, K_eta_particles_tm1), self.e_s_tph[t-1, :, :])

        prob_t_tp1h = tf.reshape(self.e_s_tph[t-1, :, :], (1, self.N, self.M))*tf.einsum("pni, pnij -> pnj", x_tm1, K_eta_particles_tm1)/tf.reshape(normalizing, (self.Nx, self.N, 1) )

        x_t = tfp.distributions.OneHotCategorical( probs = prob_t_tp1h, dtype=tf.float32).sample()

        logw_t = tf.math.log(normalizing) -tf.math.log(tf.einsum("pnj, nj -> pn", x_t, self.tildee_s_tph[t-1, :, :]))

        K_eta_particles_t = self.K_eta_multi(tf.reduce_sum(x_t, axis = 1))
        normalizing_next = tf.einsum("pnj, nj -> pn", tf.einsum("pni, pnij -> pnj", x_t, K_eta_particles_t), self.e_s_tph[t, :, :])

        return x_t, tf.reduce_sum(logw_t, axis =1), tf.reduce_sum(tf.math.log(normalizing_next), axis =1) 

    def step_0(self, Y):

        tildee_0_h = self.run_tildee_0_h(Y[   :, :, 0:(self.h+1)])

        normalizing = tf.einsum("ni, pni -> pn", tildee_0_h, self.initial_distribution)

        prob_0_1h = self.initial_distribution*tf.reshape(tildee_0_h, (1, self.N, self.M))/tf.reshape(normalizing, (self.Nx, self.N, 1) )

        x_0 = tfp.distributions.OneHotCategorical( probs = prob_0_1h, dtype=tf.float32).sample()

        logw_0        = tf.math.log(normalizing) - tf.math.log(tf.einsum("ni, pni -> pn", tildee_0_h, x_0))

        K_eta_particles_0 = self.K_eta_multi(tf.reduce_sum(x_0, axis = 1))        
        e_s_tph, _ = self.run_e_t_tph_tildee_t_tph(1, Y[   :, :, 1:(1+self.h+1)])
        normalizing_next = tf.einsum("pnj, nj -> pn", tf.einsum("pni, pnij -> pnj", x_0, K_eta_particles_0), e_s_tph)

        return x_0, tf.reduce_sum(logw_0, axis =1), tf.reduce_sum(tf.math.log(normalizing_next), axis =1) 

    def step_t(self, t, Y, x_tm1):

        K_eta_particles_tm1 = self.K_eta_multi(tf.reduce_sum(x_tm1, axis = 1)) 
        e_s_tph, tildee_s_tph = self.run_e_t_tph_tildee_t_tph(t, Y[   :, :, (t):(t+self.h+1)])
        normalizing     = tf.einsum("pnj, nj -> pn", tf.einsum("pni, pnij -> pnj", x_tm1, K_eta_particles_tm1), e_s_tph)

        prob_t_tp1h = tf.reshape(e_s_tph, (1, self.N, self.M))*tf.einsum("pni, pnij -> pnj", x_tm1, K_eta_particles_tm1)/tf.reshape(normalizing, (self.Nx, self.N, 1) )

        x_t = tfp.distributions.OneHotCategorical( probs = prob_t_tp1h, dtype=tf.float32).sample()

        logw_t = tf.math.log(normalizing) -tf.math.log(tf.einsum("pnj, nj -> pn", x_t, tildee_s_tph))

        K_eta_particles_t = self.K_eta_multi(tf.reduce_sum(x_t, axis = 1))
        e_s_tph_next, _ = self.run_e_t_tph_tildee_t_tph(t, Y[   :, :, (t+1):(t+1+self.h+1)])
        normalizing_next = tf.einsum("pnj, nj -> pn", tf.einsum("pni, pnij -> pnj", x_t, K_eta_particles_t), e_s_tph_next)

        return x_t, tf.reduce_sum(logw_t, axis =1), tf.reduce_sum(tf.math.log(normalizing_next), axis =1) 

    def step_final(self, t, Y, x_tm1):

        K_eta_particles_tm1 = self.K_eta_multi(tf.reduce_sum(x_tm1, axis = 1)) 
        e_s_tph, tildee_s_tph = self.run_e_t_tph_tildee_t_tph(t, Y[   :, :, (t):(t+self.h+1)])
        normalizing     = tf.einsum("pnj, nj -> pn", tf.einsum("pni, pnij -> pnj", x_tm1, K_eta_particles_tm1), e_s_tph)

        prob_t_tp1h = tf.reshape(e_s_tph, (1, self.N, self.M))*tf.einsum("pni, pnij -> pnj", x_tm1, K_eta_particles_tm1)/tf.reshape(normalizing, (self.Nx, self.N, 1) )

        x_t = tfp.distributions.OneHotCategorical( probs = prob_t_tp1h, dtype=tf.float32).sample()

        logw_t = tf.math.log(normalizing) -tf.math.log(tf.einsum("pnj, nj -> pn", x_t, tildee_s_tph))

        return x_t, tf.reduce_sum(logw_t, axis =1) 
    
    def log_normalize(self, logw):

        m = tf.reduce_max(logw)
        prob = tf.exp(logw-m)/tf.reduce_sum(tf.exp(logw-m))
        ESS  = tf.math.pow(tf.reduce_sum(tf.exp(logw-m)), 2)/tf.reduce_sum(tf.exp(2*(logw-m)))

        return prob, ESS

    def loglikelihood_comp(self, logw):

        m = tf.reduce_max(logw)

        return m + tf.math.log(tf.reduce_mean(tf.exp(logw-m), axis = 0))


class proposal_APF():

    def __init__(self, Nx, N, M, initial_distribution, K_eta_multi, q,):
        
        self.Nx    = Nx
        self.N     = N
        self.M     = M
        self.K_eta_multi  = K_eta_multi
        if initial_distribution.shape[0] == self.M:
            self.initial_distribution  = initial_distribution*tf.ones((self.Nx, self.N, self.M))
        elif initial_distribution.shape[0] == self.N:
            self.initial_distribution  = tf.reshape(initial_distribution, (1, N, M))*tf.ones((Nx, N, M))
            
        self.q     = q

    def step_0(self, Y):

        prob_0_1h = self.initial_distribution

        x_0 = tfp.distributions.OneHotCategorical( probs = prob_0_1h, dtype=tf.float32).sample()

        logw_0 = tf.math.log(tf.ones((self.Nx, self.N)))

        return x_0, tf.reduce_sum(logw_0, axis =1), tf.reduce_sum(tf.math.log(tf.ones((self.Nx, self.N))), axis =1) 

    def prediction(self, K_eta_particles_tm1, x_tm1):
    
        return tf.einsum("pni, pnij -> pnj", x_tm1, K_eta_particles_tm1)

    def update(self, y, pred_x_t_tm1):
    
        return tf.reshape(tf.reshape(self.q, (1, self.M))*y + (1-tf.reshape(self.q, (1, self.M)))*(1-tf.reduce_sum(y, axis = 1, keepdims = True)), (1, self.N, self.M))*pred_x_t_tm1

    def step_t(self, t, Y, x_tm1):

        K_eta_particles_tm1 = self.K_eta_multi(tf.reduce_sum(x_tm1, axis = 1)) 
        pred_x_t_tm1        = self.prediction(K_eta_particles_tm1, x_tm1)
        update_x_t          = self.update(Y[:,:,t], pred_x_t_tm1)

        normalizing     = tf.reduce_sum(update_x_t, axis =2)

        prob_t_tp1h = update_x_t/tf.reshape(normalizing, (self.Nx, self.N, 1) )

        x_t = tfp.distributions.OneHotCategorical( probs = prob_t_tp1h, dtype=tf.float32).sample()

        logw_t = tf.math.log(normalizing) 

        return x_t, tf.reduce_sum(logw_t, axis =1), tf.reduce_sum(tf.math.log(tf.ones((self.Nx, self.N))), axis =1) 


    def step_final(self, t, Y, x_tm1):
    
        K_eta_particles_tm1 = self.K_eta_multi(tf.reduce_sum(x_tm1, axis = 1)) 
        pred_x_t_tm1        = self.prediction(K_eta_particles_tm1, x_tm1)
        update_x_t          = self.update(Y[:,:,t], pred_x_t_tm1)

        normalizing     = tf.reduce_sum(update_x_t, axis =2)

        prob_t_tp1h = update_x_t/tf.reshape(normalizing, (self.Nx, self.N, 1) )

        x_t = tfp.distributions.OneHotCategorical( probs = prob_t_tp1h, dtype=tf.float32).sample()

        logw_t = tf.math.log(normalizing) 

        return x_t, tf.reduce_sum(logw_t, axis =1)
    
    def log_normalize(self, logw):

        m = tf.reduce_max(logw)
        prob = tf.exp(logw-m)/tf.reduce_sum(tf.exp(logw-m))
        ESS  = tf.math.pow(tf.reduce_sum(tf.exp(logw-m)), 2)/tf.reduce_sum(tf.exp(2*(logw-m)))

        return prob, ESS

    def loglikelihood_comp(self, logw):

        m = tf.reduce_max(logw)

        return m + tf.math.log(tf.reduce_mean(tf.exp(logw-m), axis = 0))


class proposal_BPF():

    def __init__(self, Nx, N, M, initial_distribution, K_eta_multi, q,):
        
        self.Nx    = Nx
        self.N     = N
        self.M     = M
        self.K_eta_multi  = K_eta_multi
        if initial_distribution.shape[0] == self.M:
            self.initial_distribution  = initial_distribution*tf.ones((self.Nx, self.N, self.M))
        elif initial_distribution.shape[0] == self.N:
            self.initial_distribution  = tf.reshape(initial_distribution, (1, N, M))*tf.ones((Nx, N, M))
            
        self.q     = q

    def step_0(self, Y):

        prob_0_1h = self.initial_distribution

        x_0 = tfp.distributions.OneHotCategorical( probs = prob_0_1h, dtype=tf.float32).sample()

        logw_0 = tf.math.log(tf.ones((self.Nx, self.N)))

        return x_0, tf.reduce_sum(logw_0, axis =1), tf.reduce_sum(tf.math.log(tf.ones((self.Nx, self.N))), axis =1) 

    def prediction(self, K_eta_particles_tm1, x_tm1):
    
        return tf.einsum("pni, pnij -> pnj", x_tm1, K_eta_particles_tm1)

    def update(self, y, pred_x_t_tm1):
    
        return tf.reshape(tf.reshape(self.q, (1, self.M))*y + (1-tf.reshape(self.q, (1, self.M)))*(1-tf.reduce_sum(y, axis = 1, keepdims = True)), (1, self.N, self.M))*pred_x_t_tm1

    def step_t(self, t, Y, x_tm1):

        K_eta_particles_tm1 = self.K_eta_multi(tf.reduce_sum(x_tm1, axis = 1)) 
        pred_x_t_tm1        = self.prediction(K_eta_particles_tm1, x_tm1)

        prob_t_tp1h = pred_x_t_tm1

        x_t = tfp.distributions.OneHotCategorical( probs = prob_t_tp1h, dtype=tf.float32).sample()

        logw_t = tf.math.log(tf.reduce_sum(self.update(Y[:,:,t], x_t), axis = 2))

        return x_t, tf.reduce_sum(logw_t, axis =1), tf.reduce_sum(tf.math.log(tf.ones((self.Nx, self.N))), axis =1) 


    def step_final(self, t, Y, x_tm1):
    
        K_eta_particles_tm1 = self.K_eta_multi(tf.reduce_sum(x_tm1, axis = 1)) 
        pred_x_t_tm1        = self.prediction(K_eta_particles_tm1, x_tm1)
        update_x_t          = self.update(Y[:,:,t], pred_x_t_tm1)

        normalizing     = tf.reduce_sum(update_x_t, axis =2)

        prob_t_tp1h = update_x_t/tf.reshape(normalizing, (self.Nx, self.N, 1) )

        x_t = tfp.distributions.OneHotCategorical( probs = prob_t_tp1h, dtype=tf.float32).sample()

        logw_t = tf.math.log(normalizing) 

        return x_t, tf.reduce_sum(logw_t, axis =1)
    
    def log_normalize(self, logw):

        m = tf.reduce_max(logw)
        prob = tf.exp(logw-m)/tf.reduce_sum(tf.exp(logw-m))
        ESS  = tf.math.pow(tf.reduce_sum(tf.exp(logw-m)), 2)/tf.reduce_sum(tf.exp(2*(logw-m)))

        return prob, ESS

    def loglikelihood_comp(self, logw):

        m = tf.reduce_max(logw)

        return m + tf.math.log(tf.reduce_mean(tf.exp(logw-m), axis = 0))