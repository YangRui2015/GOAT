import os

import numpy as np
from mpi4py import MPI
import time
import copy

from goat.common import logger
from goat.common import tf_util
from goat.common.util import set_global_seeds
from goat.common.mpi_moments import mpi_moments
import goat.algo.config as config
from goat.algo.rollout import RolloutWorker
from goat.algo.util import dump_params

def mpi_average(value):
    if not isinstance(value, list):
        value = [value]
    if not any(value):
        value = [0.]
    return mpi_moments(np.array(value))[0]


def train(*, policy, rollout_worker, evaluator,
          n_epochs, n_test_rollouts, n_cycles, n_batches, policy_save_interval,
          save_path, random_init, play_no_training, offline_train, **kwargs):
    rank = MPI.COMM_WORLD.Get_rank()

    if save_path and not play_no_training:
        latest_policy_path = os.path.join(save_path, 'policy_latest.pkl')
        best_policy_path = os.path.join(save_path, 'policy_best.pkl')
        periodic_policy_path = os.path.join(save_path, 'policy_{}.pkl')

    # random_init for o/g/rnd stat and model training
    if random_init and not play_no_training and not offline_train:
        logger.info('Random initializing ...')
        rollout_worker.clear_history()
        for epi in range(int(random_init) // rollout_worker.rollout_batch_size): 
            episode = rollout_worker.generate_rollouts(random_ac=True)
            policy.store_episode(episode)

    best_success_rate = -1
    logger.info('Start training...')
    for epoch in range(n_epochs):
        time_start = time.time()
        rollout_worker.clear_history()
        critic_loss_list, actor_loss_list = [], []
        for i in range(n_cycles):
            policy.dynamic_batch = False
            if not offline_train:
                episode = rollout_worker.generate_rollouts()
                policy.store_episode(episode)
            for _ in range(n_batches):   
                critic_loss, actor_loss = policy.train()
                critic_loss_list.append(critic_loss)
                actor_loss_list.append(actor_loss)
            policy.update_target_net()

        # test
        evaluator.clear_history()
        evaluator.render = True
        for _ in range(n_test_rollouts):
            evaluator.generate_rollouts()

        # record logs
        time_end = time.time()
        logger.record_tabular('epoch', epoch)
        logger.record_tabular('epoch time(min)', (time_end - time_start)/60)
        if not offline_train: # In the offline setting, we don't need to collect data
            for key, val in rollout_worker.logs('train'):
                logger.record_tabular(key, mpi_average(val))

        for key, val in policy.logs():
            logger.record_tabular(key, mpi_average(val))
        for key, val in evaluator.logs('test'):
            logger.record_tabular(key, mpi_average(val))
        if policy.use_supervised:
            logger.record_tabular('train/critic loss', mpi_average(np.mean(critic_loss_list)))
            logger.record_tabular('train/supervised loss', mpi_average(np.mean(policy.supervised_loss)))
            policy.supervised_loss = []
        
        if rank == 0:
            logger.dump_tabular()

        # save the policy if it's better than the previous ones
        success_rate = mpi_average(evaluator.current_success_rate())
        if rank == 0 and success_rate > best_success_rate and save_path and not play_no_training:
            best_success_rate = success_rate
            logger.info('New best success rate: {}. Saving policy to {} ...'.format(best_success_rate, best_policy_path))
            evaluator.save_policy(best_policy_path)
            evaluator.save_policy(latest_policy_path)
        if rank == 0 and policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_path and not play_no_training:
            policy_path = periodic_policy_path.format(epoch)
            logger.info('Saving periodic policy to {} ...'.format(policy_path))
            evaluator.save_policy(policy_path)

        # make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]

    return policy


def learn(*, env, num_epoch, 
    seed=None,
    eval_env=None,
    replay_strategy='future',
    policy_save_interval=0,
    clip_return=True,
    no_relabel=False,
    demo_file=None,
    override_params=None,
    reuse_graph=False,
    load_model=False,
    load_buffer=False,
    load_path=None,
    save_path=None,
    no_log_params=False,
    play_no_training=False,
    offline_train=False,
    mode=None,
    su_method='',
    **kwargs
):
    override_params = override_params or {} 
    rank = MPI.COMM_WORLD.Get_rank()
    if MPI is not None:
        rank = MPI.COMM_WORLD.Get_rank()
        num_cpu = MPI.COMM_WORLD.Get_size()

    # Seed everything.
    rank_seed = seed + 1000000 * rank if seed is not None else None
    set_global_seeds(rank_seed)

    # Prepare params.
    params = copy.deepcopy(config.DEFAULT_PARAMS)
    env_name = env.spec.id

    params['env_name'] = env_name
    params['replay_strategy'] = replay_strategy
    # merge env-specific parameters for offline training
    if env_name in config.DEFAULT_ENV_PARAMS and offline_train:
        params.update(config.DEFAULT_ENV_PARAMS[env_name])  
        
    params.update(**override_params)  # makes it possible to override any parameter
    params.update(kwargs)   # make kwargs part of params
    if 'num_epoch' in params:
        num_epoch = params['num_epoch']
    params['no_relabel'] = no_relabel
    params['mode'] = mode
    params['su_method'] = su_method
    params = config.prepare_params(params)
    params['rollout_batch_size'] = env.num_envs
    random_init = params['random_init']

    if env_name == 'FetchSlide-v1' and 'right' in load_path.split('/')[-1]:
        params['wgcsl_params']['relabel_ratio'] = 0.2
    elif env_name == 'FetchSlide-v1' and 'far' in load_path.split('/')[-1]:
        params['wgcsl_params']['relabel_ratio'] = 0.5

    # save total params
    if not no_log_params:
        dump_params(logger, params)
        if rank == 0:
            config.log_params(params, logger=logger)

    dims = config.configure_dims(params)
    policy = config.configure_agent(dims=dims, params=params, clip_return=clip_return, offline_train=offline_train, reuse=reuse_graph)
    if load_path is not None:
        if load_model:
            path = load_path if '.pkl' in load_path else os.path.join(load_path, 'policy_last.pkl')
            tf_util.load_variables(path)
        
        if load_buffer:
            path = load_path if '.pkl' in load_path else os.path.join(load_path, 'buffer.pkl')
            policy.buffer.load(path)
            

    rollout_params = {
        'exploit': False,
        'use_target_net': False,
        'use_demo_states': True,
        'compute_Q': False,
        'T': params['T'],
    }
    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'use_demo_states': False,
        'compute_Q': True,
        'T': params['T'],
    }
    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]

    eval_env = eval_env or env
    rollout_worker = RolloutWorker(env, policy, dims, logger, monitor=True, **rollout_params)
    evaluator = RolloutWorker(eval_env, policy, dims, logger, **eval_params)

    # no training
    if play_no_training:  
        # sample trajetories
        policy.buffer.clear_buffer()
        if 'Point' in env_name:
            num_trajs = 1
            num_episode = len(evaluator.venv.envs[0].fixed_goal_set) * num_trajs
        else:
            num_episode = 200
        
        ## for exploration, not used for evaluation
        # evaluator.noise_eps = 0.1
        # evaluator.exploit = False
        curr_trajs = 0
        success = 0
        sum_return = 0
        for _ in range(num_episode):
            if 'Point' in env_name:
                g = np.array([evaluator.venv.envs[0].fixed_goal_set[curr_trajs // num_trajs] ])
                episode = evaluator.generate_rollouts(assign_goal=g)
            else:
                episode = evaluator.generate_rollouts()
            policy.store_episode(episode, update_stats=False)
            success += episode['r'][0][-1] + 1
            sum_return += (episode['r']+1).sum() 
            curr_trajs += 1
        logger.log('success rate: ', success / curr_trajs)
        logger.log('average return: ', sum_return / curr_trajs)
        if 'Point' in env_name and save_path:
            buffer_path = os.path.join(save_path, 'buffer_{}.pkl'.format(env_name))
            policy.buffer.save(buffer_path)
        return [success / curr_trajs, sum_return / curr_trajs]

    return train(
        save_path=save_path, policy=policy, rollout_worker=rollout_worker,
        evaluator=evaluator, n_epochs=num_epoch, n_test_rollouts=params['n_test_rollouts'],
        n_cycles=params['n_cycles'], n_batches=params['n_batches'],
        policy_save_interval=policy_save_interval, demo_file=demo_file, random_init=random_init,
        play_no_training=play_no_training, offline_train=offline_train)

