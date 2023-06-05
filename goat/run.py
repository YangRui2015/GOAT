import sys
import re
import os
import os.path as osp
import gym
import copy
import tensorflow as tf
import numpy as np

from goat.common.env_util import get_env_type, build_env, get_game_envs
from goat.common import logger
from goat.common.parse_args import arg_parser
from goat.algo.train import learn
from goat.util import init_logger
from goat.algo.config import get_weight_params
from goat.envs.env_util import get_OOD_envs


_game_envs = get_game_envs()

def train(args):
    env_type, env_id = get_env_type(args, _game_envs)
    seed = args.seed
    alg_kwargs = {}
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
    parser = arg_parser()
    parser.add_argument('--env', help='environment name', type=str, default='FetchReach')
    parser.add_argument('--env_type', help='type of environment, used when the environment type cannot be automatically determined', type=str)
    parser.add_argument('--seed', help='random seed', type=int, default=None)
    # parser.add_argument('--network', help='network type mlp', default='mlp')
    parser.add_argument('--num_epoch', help='number of epochs to train', type=int, default=50)
    parser.add_argument('--random_init', help='number of random samples (used for online setting)', type=int, default=0)
    parser.add_argument('--num_env', help='Number of environment copies being run', default=1, type=int)
    parser.add_argument('--reuse_graph', help='If reuse the graph for multiple evaluation', action='store_true', default=False)
    parser.add_argument('--no_relabel', help='if not use relabel',  action='store_true', default=False)
    parser.add_argument('--play', default=False, action='store_true', help='evalutation after training')
    parser.add_argument('--play_no_training', default=False, action='store_true', help='evaluation without training')
    parser.add_argument('--mode', help='mode of algorithms "supervised"', default=None, type=str)
    parser.add_argument('--su_method', help='method for supervised learning', default='', type=str)
    parser.add_argument('--offline_train', help='If training offline or not', default=False, action='store_true')    
    # for log
    parser.add_argument('--save_path', help='Path to save trained model to', default=None, type=str)
    parser.add_argument('--load_path', help='Path to load trained model to', default=None, type=str)
    parser.add_argument('--log_path', help='Directory to save learning curve data.', default=None, type=str)
    parser.add_argument('--no_log_params', default=False, action='store_true')
    parser.add_argument('--no_init_logger', default=False, action='store_true')
    parser.add_argument('--save_buffer', help='If save the buffer or not', action='store_true')
    parser.add_argument('--load_buffer', help='If to load the offline buffer', action='store_true')
    parser.add_argument('--load_model', help='If to load the saved model', action='store_true')

    # weight params for uncertainty weight
    parser.add_argument('--weight_ratio',  default=1.5, type=float)
    parser.add_argument('--weight_min', default=0.5, type=float)
    parser.add_argument('--weight_max',  default=1, type=float)
    # expectile regression
    parser.add_argument('--ER_tau', default=0.5, type=float)
    args = parser.parse_args()
    ##### training
    main(args)

    # ## OOD eval after training
    # Do not need this for online fine-tuning
    if args.offline_train:
        load_paths = [args.save_path]
        envs = get_OOD_envs(args.env, args.load_path)
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
                # Point tasks 
                if 'Point' in env:
                    temp_args.__setattr__('save_buffer', True)  
                else:
                    temp_args.__setattr__('save_path', None)        
                    temp_args.__setattr__('save_buffer', False) 
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
