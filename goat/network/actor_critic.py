import tensorflow as tf
from goat.algo.util import store_args, nn, bi_nn

class ActorCritic: 
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers, use_bilinear=False, **kwargs):
        """The actor-critic network and related training code.
        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            o_stats (goat.algo.Normalizer): normalizer for observations
            g_stats (goat.algo.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """
        ## if use bilinear network, replace the Q networks with bi_nn, 
        # but the bilinear network is not used in our experiments because we found it does not work well
        create_nn = nn if not self.use_bilinear else bi_nn
        policy_layers = [self.hidden] * self.layers + [self.dimu]
        Q_layers = [self.hidden] * self.layers + [1] if not self.use_bilinear else [self.hidden] * self.layers + [3]

        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u'] 

        if self.use_conservation and self.net_type=='main':
            self.negative_u_tf = inputs_tf['neg_u']

        # Prepare inputs for actor and critic.
        o = self.o_stats.normalize(self.o_tf)
        g = self.g_stats.normalize(self.g_tf)
        input_pi = tf.concat(axis=1, values=[o, g])  

        # Networks.
        with tf.variable_scope('pi'):
            self.pi_tf = self.max_u * tf.tanh(nn(input_pi, policy_layers, name=self.net_type+'/pi'))
   
        with tf.variable_scope('Q'):
            # for policy training
            if not self.use_bilinear:
                input_Q = tf.concat(axis=1, values=[o, g, self.pi_tf / self.max_u])
                self.Q_pi_tf = create_nn(input_Q, Q_layers, name=self.net_type+'/Q')
            else:
                input_Q_sa = tf.concat(axis=1, values=[o, self.pi_tf / self.max_u])
                input_Q_sg = tf.concat(axis=1, values=[o, g])
                self.Q_pi_tf = create_nn(input_Q_sa, input_Q_sg, Q_layers, name=self.net_type+'/Q')

            # for critic training
            if not self.use_bilinear:
                input_Q = tf.concat(axis=1, values=[o, g, self.u_tf / self.max_u])
                self._input_Q = input_Q  # exposed for tests
                self.Q_tf = create_nn(input_Q, Q_layers, reuse=True, name=self.net_type+'/Q') 
            else:
                input_Q_sa = tf.concat(axis=1, values=[o, self.u_tf / self.max_u])
                self.Q_tf = create_nn(input_Q_sa, input_Q_sg, Q_layers, reuse=True, name=self.net_type+'/Q') 

             #### Q value 
            if self.use_conservation and self.net_type=='main':
                if not self.use_bilinear:
                    input_Q = tf.concat(axis=1, values=[o, g, self.negative_u_tf / self.max_u])
                    self.Q_tf_neg = create_nn(input_Q, Q_layers, reuse=True, name=self.net_type+'/Q') 
                else:
                    input_Q_sa = tf.concat(axis=1, values=[o, self.negative_u_tf / self.max_u])
                    self.Q_tf_neg = create_nn(input_Q_sa, input_Q_sg, Q_layers, reuse=True, name=self.net_type+'/Q') 
            



    




