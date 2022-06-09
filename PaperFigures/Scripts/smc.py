import tensorflow as tf
import tensorflow_probability as tfp

class smc():
    
    def __init__(self, proposal_distribution):

        self.proposal_distribution = proposal_distribution

    def run(self, Y):

        T = Y.shape[2]

        x_t, logw_t, logcorrection = self.proposal_distribution.step_0(Y)

        logtildew_tm1 = logw_t

        pt = logtildew_tm1 + logcorrection
        pt, ESSt = self.proposal_distribution.log_normalize(pt)
        ESS_history = tf.reshape((100*ESSt/self.proposal_distribution.Nx), (1,))

        logtildew_tm1 = tf.math.log(self.proposal_distribution.log_normalize(logtildew_tm1)[0])

        X = tf.reshape(tf.reduce_sum(x_t, axis =1), (self.proposal_distribution.Nx, self.proposal_distribution.M, 1))

        loglikelihood = []

        for t in range(1, T-1):

            indeces = tfp.distributions.Categorical(probs = pt).sample(self.proposal_distribution.Nx)
            x_tm1         = tf.gather(x_t,           indeces, axis = 0)
            logtildew_tm1 = tf.gather(logtildew_tm1, indeces, axis = 0)
            pt            = tf.gather(pt, indeces, axis = 0)

            logtildew_tm1 = logtildew_tm1 - tf.math.log(pt)

            x_t, logw_t, logcorrection = self.proposal_distribution.step_t(t, Y, x_tm1)

            logtildew_tm1 = logtildew_tm1 + logw_t

            loglikelihood.append(self.proposal_distribution.loglikelihood_comp(logtildew_tm1))

            logtildew_tm1 = tf.math.log(self.proposal_distribution.log_normalize(logtildew_tm1)[0])

            pt = logtildew_tm1 + logcorrection
            pt, ESSt = self.proposal_distribution.log_normalize(pt)
            ESS_history = tf.concat((ESS_history, tf.reshape((100*ESSt/self.proposal_distribution.Nx), (1,))), axis = 0)
            
            X = tf.concat((X, tf.reshape(tf.reduce_sum(x_t, axis =1), (self.proposal_distribution.Nx, self.proposal_distribution.M, 1))), axis = 2)

        t = t +1
        indeces = tfp.distributions.Categorical(probs = pt).sample(self.proposal_distribution.Nx)
        x_tm1         = tf.gather(x_t,           indeces, axis = 0)
        logtildew_tm1 = tf.gather(logtildew_tm1, indeces, axis = 0)
        pt            = tf.gather(pt, indeces, axis = 0)

        logtildew_tm1 = logtildew_tm1 - tf.math.log(pt)

        x_t, logw_t = self.proposal_distribution.step_final(t, Y, x_tm1)

        logtildew_tm1 = logtildew_tm1 + logw_t

        loglikelihood.append(self.proposal_distribution.loglikelihood_comp(logtildew_tm1))

        logtildew_tm1 = tf.math.log(self.proposal_distribution.log_normalize(logtildew_tm1)[0])

        pt = logtildew_tm1
        pt, ESSt = self.proposal_distribution.log_normalize(pt)
        ESS_history = tf.concat((ESS_history, tf.reshape((100*ESSt/self.proposal_distribution.Nx), (1,))), axis = 0)
        
        X = tf.concat((X, tf.reshape(tf.reduce_sum(x_t, axis =1), (self.proposal_distribution.Nx, self.proposal_distribution.M, 1))), axis = 2)

        return X, ESS_history, loglikelihood

