import tensorflow as tf
from wgcsl.algo.util import store_args, nn

class ActorCritic: 
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers, **kwargs):
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
        if self.use_vex and self.net_type=='main':
            num = 5
            self.g_vex_lis = [tf.placeholder(tf.float32, shape=(None, self.dimg)) for _ in range(num)]
            self.g_vex_norm_lis = [self.g_stats.normalize(x) for x in self.g_vex_lis]

        if self.use_conservation and self.net_type=='main':
            self.negative_u_tf = inputs_tf['neg_u']

        # Prepare inputs for actor and critic.
        o = self.o_stats.normalize(self.o_tf)
        g = self.g_stats.normalize(self.g_tf)

        ###
        input_pi = tf.concat(axis=1, values=[o, g, g - o[:,3:6]])  # for actor
        # input_pi = g - o  # oracle

        # Networks.
        with tf.variable_scope('pi'):
            self.pi_tf = self.max_u * tf.tanh(nn(
                input_pi, [self.hidden] * self.layers + [self.dimu]))
            if self.use_noise_p and self.net_type=='main':
                noise_input = tf.concat(axis=1, values=[noise_o, noise_g, noise_g - noise_o[:,3:6]]) 
                # noise_input = noise_g - noise_o  # oracle
                self.noise_pi_tf = self.max_u * tf.tanh(nn(
                    noise_input, [self.hidden] * self.layers + [self.dimu], reuse=True))

        with tf.variable_scope('Q'):
            # for policy training
            input_Q = tf.concat(axis=1, values=[o, g, g - o[:,3:6], self.pi_tf / self.max_u])
            # input_Q = tf.concat(axis=1, values=[g - o, self.pi_tf / self.max_u])  # oracle
            self.Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1])
            # for critic training
            input_Q = tf.concat(axis=1, values=[o, g, g-o[:,3:6], self.u_tf / self.max_u])
            # input_Q = tf.concat(axis=1, values=[g - o, self.u_tf / self.max_u]) # oracle
            self._input_Q = input_Q  # exposed for tests
            self.Q_tf = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True) 

            #### Q value 
            if self.use_conservation and self.net_type=='main':
                input_Q = tf.concat(axis=1, values=[o, g, g-o[:,3:6], self.negative_u_tf / self.max_u])
                # input_Q = tf.concat(axis=1, values=[g - o, self.negative_u_tf / self.max_u]) # oracle
                self.Q_tf_neg = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True) 

            if self.use_noise_q and self.net_type=='main':
                noise_input_a = tf.concat(axis=1, values=[noise_o, noise_g, noise_g-noise_o[:,3:6], self.u_tf / self.max_u]) 
                # noise_input_a = tf.concat(axis=1, values=[noise_g - noise_o, self.u_tf / self.max_u]) # oracle
                self.noise_Q_tf = nn(noise_input_a, [self.hidden] * self.layers + [1], reuse=True) 

        if self.use_vex and self.net_type == 'main':
            self.pi_vex_tf = []
            for g_tmp in self.g_vex_norm_lis:
                input_tmp = tf.concat(axis=1, values=[o, g_tmp, g_tmp - o[:,3:6]])
                # input_tmp = tf.concat(axis=1, values=[g_tmp - o]) # oracle
                with tf.variable_scope('pi'):
                    self.pi_vex_tf.append(self.max_u * tf.tanh(nn(input_tmp, [self.hidden] * self.layers + [self.dimu], reuse=True)))

                    


    
    
            


    



