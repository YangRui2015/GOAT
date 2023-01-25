from wgcsl.common import logger
import numpy as np
from wgcsl.algo.util import random_log
from wgcsl.algo.adv_que import advque, vque, stdque

global global_threshold 
global global_threshold_density 
global_threshold = 0
global_threshold_density = 0
global global_std_steps 
global_std_steps = 0


def make_random_sample(reward_fun):
    def _random_sample(episode_batch, batch_size_in_transitions): 
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        T = episode_batch['u'].shape[1]    # steps of a episode
        rollout_batch_size = episode_batch['u'].shape[0]   # number of episodes
        batch_size = batch_size_in_transitions   # number of goals sample from rollout
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                            for key in episode_batch.keys()}
        # Reconstruct info dictionary for reward computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # # Re-compute reward since we may have substituted the u and o_2 ag_2
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info
        transitions['r'] = reward_fun(**reward_params)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                        for k in transitions.keys()}
        assert(transitions['u'].shape[0] == batch_size_in_transitions)

        return transitions
    return _random_sample


def make_sample_transitions(replay_strategy, replay_k, reward_fun, no_relabel=False):
    if replay_strategy == 'future':
        future_p = 1 - (1. / (1 + replay_k))
    else:  
        future_p = 0

    if no_relabel:
        print( '*' * 10 + 'Do not use relabeling in this method' + '*' * 10)
    
    def _preprocess(episode_batch, batch_size_in_transitions, p=None):
        T = episode_batch['u'].shape[1]    # steps of a episode
        rollout_batch_size = episode_batch['u'].shape[0]   # number of episodes
        batch_size = batch_size_in_transitions   # number of goals sample from rollout
        if p is not None:
            episode_idxs = np.random.choice(np.arange(rollout_batch_size), size=batch_size, p=p/p.sum())
        else:
            episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys()}
        return transitions, episode_idxs, t_samples, batch_size, T

    def _get_reward(ag_2, g):
        info = {}
        reward_params = {'ag_2':ag_2, 'g':g}
        reward_params['info'] = info
        # make rewards -1/0 --->0/1
        return reward_fun(**reward_params) + 1  

    def _get_future_ags(episode_batch, episode_idxs, t_samples, batch_size, T, future_p=future_p, return_t=False):
        her_indexes = (np.random.uniform(size=batch_size) < future_p)
        future_offset = np.random.uniform(size=batch_size) * (T-t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        if not return_t:
            return future_ag.copy(), her_indexes.copy()
        else:
            return future_ag.copy(), her_indexes.copy(), future_offset
    
        
    def _reshape_transitions(transitions, batch_size, batch_size_in_transitions):
        if 'loss' in transitions.keys():
            loss = transitions.pop('loss')
            transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}
            transitions['loss'] = loss
        else:
            transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert(transitions['u'].shape[0] == batch_size_in_transitions)
        return transitions
    
    def _sample_her_transitions(episode_batch, batch_size_in_transitions, info=None):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        transitions, episode_idxs, t_samples, batch_size, T = _preprocess(episode_batch, batch_size_in_transitions)
        if not no_relabel:
            future_ag, her_indexes = _get_future_ags(episode_batch, episode_idxs, t_samples, batch_size, T)
            if len(transitions['g'].shape) == 1:
                transitions['g'][her_indexes] = future_ag.reshape(-1)
            else:
                transitions['g'][her_indexes] = future_ag

        transitions['r'] = _get_reward(transitions['ag_2'], transitions['g'])
        return _reshape_transitions(transitions, batch_size, batch_size_in_transitions)

    def _sample_conservative_transitions(episode_batch, batch_size_in_transitions, info):
        transitions, episode_idxs, t_samples, batch_size, T = _preprocess(episode_batch, batch_size_in_transitions)
        random_action_fun, get_Q = info['random_action_fun'], info['get_Q']
        if not no_relabel:
            future_ag, her_indexes = _get_future_ags(episode_batch, episode_idxs, t_samples, batch_size, T)
            if len(transitions['g'].shape) == 1:
                transitions['g'][her_indexes] = future_ag.reshape(-1)
            else:
                transitions['g'][her_indexes] = future_ag

        actions, Qs = [], []
        negative_actions = []
        N = 20
        actions = random_action_fun(N)
        for i in range(N):
            Qs.append(get_Q(o=transitions['o'], u=np.array(actions[i]).repeat(batch_size, axis=0), g=transitions['g']))
        all_Qs = np.array(Qs).reshape((batch_size, N))
        actions = np.array(actions).reshape(-1, actions[0].shape[-1])
        for i in range(batch_size):
            exp_Qs = np.exp(all_Qs[i]) + 0.0001
            try:
                neg_act = np.random.choice(np.arange(N), p= exp_Qs / exp_Qs.sum())
            except:
                import pdb;pdb.set_trace()
            negative_actions.append(actions[neg_act])
        transitions['neg_u'] = np.array(negative_actions).reshape((batch_size, transitions['u'].shape[-1]))
        transitions['r'] = _get_reward(transitions['ag_2'], transitions['g'])
        return _reshape_transitions(transitions, batch_size, batch_size_in_transitions)
    

    def _sample_supervised_transitions(episode_batch, batch_size_in_transitions, info):
        train_policy, gamma, get_Q_pi, method = info['train_policy'], info['gamma'], info['get_Q_pi'], info['method']
        baw_delta, baw_max = info['baw_delta'], info['baw_max']
        update = info['update'] # whether to update the policy or only get the policy loss for validation
        kde = info['kde'] if 'kde' in info.keys() else None
        # get_ensemble_actions = info['ensemble']
        transitions, episode_idxs, t_samples, batch_size, T = _preprocess(episode_batch, batch_size_in_transitions)

        random_log('using supervide policy learning with method {} and no relabel {}'.format(method, no_relabel))
        original_g = transitions['g'].copy() # save to train the value function
        if not no_relabel:
            future_ag, her_indexes, offset = _get_future_ags(episode_batch, episode_idxs, t_samples, batch_size, T, future_p=1, return_t=True)
            transitions['g'][her_indexes] = future_ag
        else:
            offset = np.zeros(batch_size)

        if method == '': # do not use weights for GCSL
            loss = train_policy(o=transitions['o'], g=transitions['g'], u=transitions['u'], original_g=original_g)
        else:
            method_lis = method.split('_')
            if 'gamma' in method_lis:
                weights = pow(gamma, offset)  
            else:
                weights = np.ones(batch_size)

            if 'adv' in method_lis:
                value, value_std = get_Q_pi(o=transitions['o'], g=transitions['g'], std=True)
                value, value_std = value.reshape(-1), value_std.reshape(-1)
                # value = get_Q_pi(o=transitions['o'], g=transitions['g']).reshape(-1)
                next_value = get_Q_pi(o=transitions['o_2'], g=transitions['g']).reshape(-1)
                adv = _get_reward(transitions['ag_2'], transitions['g']) + gamma * next_value  - value
                vque.update(value)

                if 'baw' or 'baw01' in method_lis:
                    global global_threshold
                    if update:
                        advque.update(adv)
                        global_threshold = min(global_threshold + baw_delta, baw_max)
                    threshold = advque.get(global_threshold)

                if 'exp' in method_lis:  # exp weights
                    if 'clip10' in method_lis:
                        weights *= np.clip(np.exp(adv), 0, 10)
                    elif 'clip5' in method_lis:
                        weights *= np.clip(np.exp(adv), 0, 5)
                    elif 'clip1' in method_lis:
                        weights *= np.clip(np.exp(adv), 0, 1)
                    else:
                        weights *= np.exp(adv) 
                elif 'clip5' in method_lis:  # clip adv weight
                    weights *= np.clip(adv, 0, 5)
                
                if 'baw01' in method_lis:
                    positive = adv.copy()
                    positive[adv >= threshold] = 1
                    positive[adv < threshold] = 0
                    weights *= positive
                elif 'baw' in method_lis:
                    positive = adv.copy()
                    positive[adv >= threshold] = 1
                    positive[adv < threshold] = 0.05
                    weights *= positive

                # if 'dv01' in method_lis and global_threshold >= 80:  
                #     global global_v_threshold
                #     global_v_threshold -= 0.001
                #     global_v_threshold = max(80, global_v_threshold) # max(70, global_v_threshold) #0.852
                #     v_threshold = vque.get(global_v_threshold)
                #     v_weight = np.ones(weights.shape)
                #     v_weight[value >= v_threshold] = 0.05
                #     weights *= v_weight
                # elif 'dvsigmoid' in method_lis and global_threshold >= 80:
                #     v_mean, v_std = vque.mean_std()
                #     norm_value = (value - v_mean) / v_std
                #     weights /= (1/(1+np.exp(-norm_value)) + 0.5)
                # elif 'dv' in method_lis and global_threshold >= 80:
                #     v_mean, v_std = vque.mean_std()
                #     norm_value = (value - v_mean) / v_std
                #     # weights /= np.clip(np.exp(0.5 * norm_value), 0.5, 5) # 0.894
                #     # weights /= np.clip(np.exp(0.5 * norm_value), 0.5, 10)
                #     # weights /= np.clip(np.exp(0.5 * norm_value), 0.25, 5)
                #     # weights /= np.clip(np.exp(0.5 * norm_value), 0.25, 10)
                #     # weights /= np.clip(np.exp(1 * norm_value), 0.5, 5)
                #     # weights /= np.clip(np.exp(0.25 * norm_value), 0.5, 5)
                #     # weights /= np.clip(np.exp(0.5 * norm_value), 0.5, 3)
                #     # weights /= np.clip(np.exp(0.1 * norm_value), 0.5, 5)
                #     weights /= np.clip(np.exp(0.5 * norm_value), 0.5, 2)

                global global_std_steps
                global_std_steps += 1

                if 'uncertainty' in method and global_std_steps >= 10 * 400:
                    # if 'duncertainty' in method_lis:
                    #     # weights /= (1+ np.clip(value_std, 0, 1))   # 0.874 circle0.75050
                    #     # weights /= np.clip(np.exp(norm_std), 0.5, 5) # 0.82475
                    #     weights /= (0.5 + np.clip(value_std * 3, 0, 1))

                    if 'uncertainty*3clip1+0.2' in method_lis:
                        weights *= np.clip(0.2 + value_std * 3, 0.2, 1) 
                    elif 'uncertainty*3clip1' in method_lis:
                        weights *= np.clip(value_std * 3, 0.2, 1)  
                    elif 'uncertainty*3' in method_lis:
                        weights *= np.clip(value_std * 3, 0.5, 1.5)   # 0.8955 circle 0.7582
                    # elif 'dduncertainty' in method_lis:
                    #     weights /= np.clip(value_std * 3, 0.5, 1.5)   # 0.8955 circle 0.7582
                    # elif 'duncertainty' in method_lis:
                    #     weights /= (0.5+np.clip(value_std * 3, 0, 1))   
                    elif 'uncertainty' in method_lis:
                        # weights *= (1+ np.clip(value_std, 0, 1)) # std >= 0 # 0.885 circle 0.76875
                        # weights *= np.clip(np.exp(value_std) - 0.5, 0.5, 2)  # 0.861
                        # weights *= (0.5 + np.clip(value_std * 2, 0, 1)) # circle 0.8067
                        weights *= (0.5 + np.clip(value_std * 3, 0, 1))  # circle 0.80725
                    
            # if kde is not None:
            #     # samples = np.concatenate([transitions['ag'], transitions['g']], axis=1)
            #     # log_density = kde.parrallel_score_samples(samples)
            #     # density = np.clip(np.exp(log_density), 0.01, 2)
            #     # weights *= 1 / np.sqrt(density + 0.6)

            #     #### weight for state-goal pair
            #     global global_threshold_density 
            #     # low, high = global_threshold_density, min(global_threshold_density + 30, 100)
            #     # weight_threshold_low = np.percentile(info['kde_weights_compact'], low)
            #     # weight_threshold_high = np.percentile(info['kde_weights_compact'], high)
            #     # temp_weights = np.zeros(info['kde_weights_compact'].shape) + 0.2
            #     # temp_weights[np.where((info['kde_weights_compact'] > weight_threshold_low) & (info['kde_weights_compact'] < weight_threshold_high))] = 1
            #     # global_threshold_density = min(global_threshold_density + 0.002, 80)

            #     weight_threshold = np.percentile(info['kde_weights_compact'], global_threshold_density)
            #     temp_weights = info['kde_weights_compact'].copy() 
            #     temp_weights[temp_weights < weight_threshold] = 0.1
            #     temp_weights[temp_weights >= weight_threshold] = 1
            #     global_threshold_density = min(global_threshold_density + 0.002, 70)

            #     idxs = episode_idxs * T*(T+1)//2 + t_samples * (2*T-t_samples+1)// 2 + offset 
            #     compact_idxs = info['kde_weights_trans'][idxs].astype(int)
            #     weights *= temp_weights[compact_idxs]
                
                

            weighted_loss = train_policy(o=transitions['o'], g=transitions['g'], u=transitions['u'], weights=weights, update=update)  
            ### supervised loss without weights
            supervised_loss = train_policy(o=transitions['o'], g=transitions['g'], u=transitions['u'], update=False)  
            loss = [weighted_loss, supervised_loss] 

        # keep_origin_rate = 0.2
        # origin_index = (np.random.uniform(size=batch_size) < keep_origin_rate)
        # transitions['g'][origin_index] = original_g[origin_index]
        transitions['r'] = _get_reward(transitions['ag_2'], transitions['g']) 
        transitions['loss'] = loss
        return _reshape_transitions(transitions, batch_size, batch_size_in_transitions)

    return _sample_supervised_transitions, _sample_her_transitions, _sample_conservative_transitions

