import sys
import re
import os
import os.path as osp
import gym
import tensorflow as tf
import numpy as np

from wgcsl.common.vec_env import VecEnv
from wgcsl.common.env_util import get_env_type, build_env, get_game_envs
from wgcsl.common import logger
from wgcsl.common.parse_args import arg_parser, get_learn_function_defaults
from wgcsl.algo.train import learn
from wgcsl.util import init_logger
from wgcsl.algo.config import get_weight_params


_game_envs = get_game_envs()


def train(args):
    env_type, env_id = get_env_type(args, _game_envs)
    seed = args.seed
    alg_kwargs = get_learn_function_defaults('her', env_type)
    weight_params = get_weight_params(args)
    alg_kwargs.update(weight_params)
    env = build_env(args, _game_envs)
    logger.log('Training {} with {} on {}:{} '.format(args.mode, args.su_method, env_type, env_id))

    ## make save dir
    if args.save_path:
        os.makedirs(os.path.expanduser(args.save_path), exist_ok=True)

    model = learn(
        env=env,
        seed=seed,
        num_epoch=args.num_epoch,
        no_relabel=args.no_relabel,
        save_path=args.save_path,
        reuse_graph=args.reuse_graph,
        load_model=args.load_model,
        load_buffer=args.load_buffer,
        load_path=args.load_path,
        play_no_training=args.play_no_training,
        offline_train=args.offline_train,
        mode=args.mode,
        su_method=args.su_method,
        no_log_params=args.no_log_params,
        **alg_kwargs
    )
    return model, env


def main(args):
    if not args.no_init_logger:
        rank = init_logger(args)
    else:
        rank = 0
    model, env = train(args)
    if args.save_path is not None and rank == 0 and hasattr(model, 'save'):
        save_path = osp.expanduser(args.save_path)
        last_policy_path = os.path.join(save_path, 'policy_last.pkl')
        model.save(last_policy_path)
        if args.save_buffer:
            buffer_path = os.path.join(save_path, 'buffer.pkl')
            model.buffer.save(buffer_path)

    return model

if __name__ == '__main__':
    # main(sys.argv)
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='Reacher-v2')
    parser.add_argument('--env_type', help='type of environment, used when the environment type cannot be automatically determined', type=str)
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--network', help='network type mlp', default='mlp')
    parser.add_argument('--num_epoch', help='number of epochs to train', type=int, default=50)
    parser.add_argument('--random_init', help='number of random samples (used for online setting)', type=int, default=0)
    parser.add_argument('--num_env', help='Number of environment copies being run', default=1, type=int)
    parser.add_argument('--reuse_graph', help='If reuse the graph for multiple evaluation', action='store_true', default=False)
    parser.add_argument('--save_path', help='Path to save trained model to', default=None, type=str)
    parser.add_argument('--load_path', help='Path to load trained model to', default=None, type=str)
    parser.add_argument('--log_path', help='Directory to save learning curve data.', default=None, type=str)
    parser.add_argument('--no_relabel', help='if not use relabel',  action='store_true', default=False)
    # for log
    parser.add_argument('--no_log_params', default=False, action='store_true')
    parser.add_argument('--no_init_logger', default=False, action='store_true')
    parser.add_argument('--save_buffer', help='If save the buffer or not', action='store_true')
    parser.add_argument('--load_buffer', help='If to load the offline buffer', action='store_true')
    parser.add_argument('--load_model', help='If to load the saved model', action='store_true')
    parser.add_argument('--play', default=False, action='store_true', help='evalutation after training')
    parser.add_argument('--play_no_training', default=False, action='store_true', help='evaluation without training')
    parser.add_argument('--mode', help='mode of algorithms "dynamic", "supervised"', default=None, type=str)
    parser.add_argument('--su_method', help='method for supervised learning', default='', type=str)
    parser.add_argument('--offline_train', help='If training offline or not', default=False, action='store_true')
    # smooth
    parser.add_argument('--use_noise_p', help='If smooth the policy or not', default=False, action='store_true')
    parser.add_argument('--use_noise_q', help='If smooth the Q function or not', default=False, action='store_true')
    parser.add_argument('--psmooth_eps', help='The smooth range of policy', default=0.0, type=float)
    parser.add_argument('--qsmooth_eps', help='The smooth range of Q function', default=0.0, type=float)
    parser.add_argument('--psmooth_reg', help='The coefficient of policy smooth loss', default=0.0, type=float)
    parser.add_argument('--qsmooth_reg', help='The coefficient of Q function smooth loss', default=0.0, type=float)
    # vex
    parser.add_argument('--use_vex', help='If use VEx or not', default=False, action='store_true')
    # weight params
    parser.add_argument('--weight_ratio',  default=1.0, type=float)
    parser.add_argument('--weight_min', default=0.5, type=float)
    parser.add_argument('--weight_max',  default=1.5, type=float)
    args = parser.parse_args()
    # # #### train
    main(args)
    # ## OOD eval after training
    load_paths = [args.save_path]
    import copy
    if 'Point' in args.env:
        envs = ['PointFixedEnv-v1', 'PointFixedLargeEnv-v1']
    elif 'HandReach' in args.env:
        envs = ['HandReachOOD-Near-v0', 'HandReachOOD-Far-v0']
    elif 'Reach' in args.env:
        if 'far' in args.save_path:
            envs = ['FetchReachOOD-Near-v1', 'FetchReachOOD-Far-v1']
        else:
            envs = ['FetchReachOOD-Left-v1', 'FetchReachOOD-Right-v1']
    elif 'Push' in args.env:
        if 'circle' in args.save_path:
            envs = ['FetchPushOOD-Near2Near-v1', 'FetchPushOOD-Near2Far-v1', 'FetchPushOOD-Far2Near-v1', 'FetchPushOOD-Far2Far-v1']
        else:
            envs = ['FetchPushOOD-Left2Left-v1', 'FetchPushOOD-Left2Right-v1', 'FetchPushOOD-Right2Right-v1', 'FetchPushOOD-Right2Left-v1']
    elif 'Pick' in args.env:
        if 'height' in args.save_path:
            envs = ['FetchPickOOD-Low2Low-v1', 'FetchPickOOD-Low2High-v1']
        else:
            envs = ['FetchPickOOD-Left2Left-v1', 'FetchPickOOD-Left2Right-v1', 'FetchPickOOD-Right2Right-v1', 'FetchPickOOD-Right2Left-v1']
    elif 'Slide' in args.env:
        if 'far' in args.save_path:
            envs = ['FetchSlideOOD-Near2Near-v1', 'FetchSlideOOD-Near2Far-v1']
        else:
            envs = ['FetchSlideOOD-Left2Left-v1', 'FetchSlideOOD-Left2Right-v1', 'FetchSlideOOD-Right2Right-v1', 'FetchSlideOOD-Right2Left-v1']
    elif 'Stack' in args.env:
        if 'far' in args.save_path:
            envs = ['FetchStackOOD-Near2Near-v1', 'FetchStackOOD-Near2Far-v1', 'FetchStackOOD-Far2Near-v1', 'FetchStackOOD-Far2Far-v1']
        else:
            envs = ['FetchStackOOD-Right2Left-v1', 'FetchStackOOD-Right2Right-v1', 'FetchStackOOD-Left2Left-v1', 'FetchStackOOD-Left2Right-v1']
    elif 'Block' in args.env:
        envs = ['HandBlockOOD-P2P-v0', 'HandBlockOOD-P2N-v0', 'HandBlockOOD-N2P-v0', 'HandBlockOOD-N2N-v0']
    else:
        envs = ['FetchPush-v1']
    res = {}
    for i, env in enumerate(envs):
        temp_args = copy.deepcopy(args)
        temp_args.__setattr__('env', env)
        env_res = [[],[]]
        for j, path in enumerate(load_paths):
            temp_args.__setattr__('reuse_graph', True)   
            temp_args.__setattr__('load_path', path)        
            temp_args.__setattr__('load_buffer', False)      
            temp_args.__setattr__('load_model', True)   
            temp_args.__setattr__('play_no_training', True)        
            # temp_args.__setattr__('save_path', None)        
            # temp_args.__setattr__('save_buffer', False) 
            # Point tasks 
            temp_args.__setattr__('save_buffer', True)  
            temp_args.__setattr__('no_log_params', True) 
            temp_args.__setattr__('no_init_logger', True) 
            success_rate, cul_return = main(temp_args)
            env_res[0].append(success_rate)
            env_res[1].append(cul_return)
        res[env] = env_res
    
    for env in envs:
        logger.log('-------------')
        logger.log('env: {}'.format(env))
        logger.log('success rate: {} +- {}'.format(np.mean(res[env][0]), np.std(res[env][0])))
        logger.log('cumulative return: {} +- {}'.format(np.mean(res[env][1]), np.std(res[env][1])))
        logger.log('-------------')
    logger.log('total average success rate: {}'.format(np.mean([res[x][0] for x in envs])))
    logger.log('total average cumulative return: {}'.format(np.mean([res[x][1] for x in envs])))


    # ###### evaluation for multiple datasets
    # import copy
    # # envs = ['FetchPushOOD-Left2Left-v1', 'FetchPushOOD-Left2Right-v1', 'FetchPushOOD-Right2Right-v1', 'FetchPushOOD-Right2Left-v1']
    # # envs = ['FetchPushOOD-Near2Near-v1', 'FetchPushOOD-Near2Far-v1', 'FetchPushOOD-Far2Near-v1', 'FetchPushOOD-Far2Far-v1']
    # # envs = ['FetchStackOOD-Right2Left-v1', 'FetchStackOOD-Right2Right-v1', 'FetchStackOOD-Left2Left-v1', 'FetchStackOOD-Left2Right-v1']
    # # envs = ['FetchStackOOD-Near2Near-v1', 'FetchStackOOD-Near2Far-v1', 'FetchStackOOD-Far2Near-v1', 'FetchStackOOD-Far2Far-v1']
    # # envs = ['FetchReachOOD-Near-v1', 'FetchReachOOD-Far-v1']
    # envs = ['HandReachOOD-Near-v0', 'HandReachOOD-Far-v0']
    # # prefix = '/home/ryangam/logs/OOD/ensemble/circle/exp_adv_clip10_baw_sumloss/'
    # # prefix = '/home/ryangam/logs/OOD/ensemble/circle/exp_adv_clip10_baw_uncertainty*5clip0.2p2_sumloss/'
    # # prefix = '/home/ryangam/logs/OOD/ensemble_pi/exp_adv_clip10_baw_trueactstd_first/' 
    # # prefix = '/home/ryangam/logs/final/ensemble_stack_far10000/exp_adv_2_clip10/'
    # # prefix = '/home/ryangam/logs/final/ensemble_reach_far_200/exp_adv_2_clip10_baw_tanhstd_nodelay_norm01_ratio2clipmax1/'
    # prefix = '/home/ryangam/logs/final/ensemble_handreach/exp_adv_2_clip10_baw_tanhstd_nodelay_norm01_ratio1clipmax1/'
    # postfix = '/HandReach-expert/policy_last.pkl'
    # # load_paths = ['WGCSL_densityexp1', 'WGCSL_densityexp2', 'WGCSL_densityexp3', 'WGCSL_densityexp4', 'WGCSL_densityexp5']
    # # load_paths = ['uncertainty_WGCSL_1', 'uncertainty_WGCSL_2', 'uncertainty_WGCSL_3', 'uncertainty_WGCSL_4']
    # # load_paths = ['WGCSL_densityselect1', 'WGCSL_densityselect2', 'WGCSL_densityselect3', 'WGCSL_densityselect4', 'WGCSL_densityselect5']
    # # load_paths = ['wgcslclip0_5_fullrelabel_1', 'wgcslclip0_5_fullrelabel_2', 'wgcslclip0_5_fullrelabel_3', 'wgcslclip0_5_fullrelabel_4', 'wgcslclip0_5_fullrelabel_5']
    # # load_paths = ['wgcslclip0_10_fullrelabel_1', 'wgcslclip0_10_fullrelabel_2', 'wgcslclip0_10_fullrelabel_3', 'wgcslclip0_10_fullrelabel_4', 'wgcslclip0_10_fullrelabel_5']
    # # load_paths = ['gcsl_1', 'gcsl_2', 'gcsl_3', 'gcsl_4', 'gcsl_5']
    # load_paths = ['seed_1', 'seed_2', 'seed_3', 'seed_4', 'seed_5']
    # load_paths = [prefix + x + postfix for x in load_paths]
    
    # res = {}
    # for i, env in enumerate(envs):
    #     temp_args = copy.deepcopy(args)
    #     temp_args.__setattr__('env', env)
    #     env_res = [[],[]]
    #     for j, path in enumerate(load_paths):
    #         if i or j:
    #             temp_args.__setattr__('reuse_graph', True)   
    #         temp_args.__setattr__('load_path', path)        
    #         temp_args.__setattr__('load_model', True) 
    #         temp_args.__setattr__('log_params', False) 
    #         success_rate, cul_return = main(temp_args)
    #         env_res[0].append(success_rate)
    #         env_res[1].append(cul_return)
    #     res[env] = env_res
    
    # print(prefix)
    # for env in envs:
    #     logger.log('-------------')
    #     logger.log('env: {}'.format(env))
    #     logger.log('success rate: {} +- {}'.format(np.mean(res[env][0]), np.std(res[env][0])))
    #     logger.log('cumulative return: {} +- {}'.format(np.mean(res[env][1]), np.std(res[env][1])))
    #     logger.log('-------------')
    # logger.log('total average success rate: {}'.format(np.mean([res[x][0] for x in envs])))
    # logger.log('total average cumulative return: {}'.format(np.mean([res[x][1] for x in envs])))
    # import pdb;pdb.set_trace()
