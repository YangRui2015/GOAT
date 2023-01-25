from email import policy
import tensorflow as tf
from wgcsl.algo.util import store_args, nn, nn_ensemble, nn_single

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
            o_stats (wgcsl.algo.Normalizer): normalizer for observations
            g_stats (wgcsl.algo.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """
        create_nn = nn_ensemble
        policy_layers = [self.hidden] * self.layers + [self.dimu]
        Q_layers = [self.hidden] * self.layers + [1] 

        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u'] 
        ### noise for training smooth policy
        if 'noise_o' in inputs_tf.keys() and self.net_type=='main':
            self.noise_o = inputs_tf['noise_o']
            noise_o = self.o_stats.normalize(self.noise_o)
        if 'noise_g' in inputs_tf.keys() and self.net_type=='main':
            self.noise_g = inputs_tf['noise_g']
            noise_g = self.g_stats.normalize(self.noise_g)

        if self.use_conservation and self.net_type=='main':
            self.negative_u_tf = inputs_tf['neg_u']

        # Prepare inputs for actor and critic.
        o = self.o_stats.normalize(self.o_tf)
        g = self.g_stats.normalize(self.g_tf)
        input_pi = tf.concat(axis=1, values=[o, g])  # for actor

        # Networks.
        with tf.variable_scope('pi'):
            self.pi_tf = self.max_u * tf.tanh(nn(input_pi, policy_layers, name=self.net_type+'/pi'))
            # self.pi_tf_list = nn_ensemble(input_pi, policy_layers, name=self.net_type+'/pi')
            # self.pi_tf_list = [self.max_u * tf.tanh(x) for x in self.pi_tf_list]
            # self.pi_tf = tf.reduce_mean(self.pi_tf_list, axis=0)
            # self.pi_tf = self.pi_tf_list[0]
            # self.pi_tf_std = tf.reduce_sum(tf.math.reduce_std(self.pi_tf_list, axis=0), axis=1)

            # if self.use_noise_p and self.net_type=='main':
            #     noise_input = tf.concat(axis=1, values=[noise_o, noise_g])
            #     self.noise_pi_tf = self.max_u * tf.tanh(tf.reduce_mean(nn_single(noise_input, policy_layers, reuse=True, name=self.net_type+'/pi'), axis=0))

        with tf.variable_scope('Q'):
            # for policy training
            input_Q = tf.concat(axis=1, values=[o, g, self.pi_tf / self.max_u])
            # self.Q_pi_tf= create_nn(input_Q, Q_layers, name=self.net_type+'/Q')
            self.Q_pi_tf_list = create_nn(input_Q, Q_layers, name=self.net_type+'/Q')
            self.Q_pi_tf = tf.reduce_mean(self.Q_pi_tf_list, axis=0)
            self.Q_pi_tf_std = tf.math.reduce_std(self.Q_pi_tf_list, axis=0)

            # for critic training
            input_Q = tf.concat(axis=1, values=[o, g, self.u_tf / self.max_u])
            # self.Q_tf = create_nn(input_Q, Q_layers, reuse=True, name=self.net_type+'/Q') 
            self.Q_tf_list = create_nn(input_Q, Q_layers, reuse=True, name=self.net_type+'/Q') 
            self.Q_tf = tf.reduce_mean(self.Q_tf_list, axis=0)
            # self.Q_tf_std = tf.math.reduce_std(self.Q_tf_list, axis=0)

             #### Q value 
            if self.use_conservation and self.net_type=='main':
                input_Q = tf.concat(axis=1, values=[o, g, self.negative_u_tf / self.max_u])
                self.Q_tf_neg_list = create_nn(input_Q, Q_layers, reuse=True, name=self.net_type+'/Q') 
                self.Q_tf_neg = tf.reduce_mean(self.Q_tf_neg_list, axis=0)
                self.Q_tf_neg_std = tf.math.reduce_std(self.Q_tf_neg_list, axis=0)

            if self.use_noise_q and self.net_type=='main':
                noise_input_a = tf.concat(axis=1, values=[noise_o, noise_g, self.u_tf / self.max_u]) 
                self.noise_Q_tf_list = create_nn(noise_input_a, Q_layers, reuse=True, name=self.net_type+'/Q') 
                self.noise_Q_tf = tf.reduce_mean(self.noise_Q_tf_list, axis=0)
                self.noise_Q_tf_std = tf.math.reduce_std(self.noise_Q_tf_list, axis=0)



    



