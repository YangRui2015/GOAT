import os

from collections import OrderedDict
from numbers import Number

import numpy as np
from gym.spaces import Box

ENV_ASSET_DIR = os.path.join(os.path.dirname(__file__), 'assets')


def get_OOD_envs(env, data_type):
    if 'Point' in env:
        envs = ['PointFixedEnv-v1', 'PointFixedLargeEnv-v1']
    elif 'HandReach' in env:
        envs = ['HandReachOOD-Near-v0', 'HandReachOOD-Far-v0']
    elif 'Reach' in env:
        if 'far' in data_type:
            envs = ['FetchReachOOD-Near-v1', 'FetchReachOOD-Far-v1']
        else:
            envs = ['FetchReachOOD-Left-v1', 'FetchReachOOD-Right-v1']
    elif 'Push' in env:
        if 'circle' in data_type:
            envs = ['FetchPushOOD-Near2Near-v1', 'FetchPushOOD-Near2Far-v1', 'FetchPushOOD-Far2Near-v1', 'FetchPushOOD-Far2Far-v1']
        else:
            envs = ['FetchPushOOD-Left2Left-v1', 'FetchPushOOD-Left2Right-v1', 'FetchPushOOD-Right2Right-v1', 'FetchPushOOD-Right2Left-v1']
    elif 'Pick' in env:
        if 'height' in data_type:
            envs = ['FetchPickOOD-Low2Low-v1', 'FetchPickOOD-Low2High-v1']
        else:
            envs = ['FetchPickOOD-Left2Left-v1', 'FetchPickOOD-Left2Right-v1', 'FetchPickOOD-Right2Right-v1', 'FetchPickOOD-Right2Left-v1']
    elif 'Slide' in env:
        if 'far' in data_type:
            envs = ['FetchSlideOOD-Near2Near-v1', 'FetchSlideOOD-Near2Far-v1']
        else:
            envs = ['FetchSlideOOD-Left2Left-v1', 'FetchSlideOOD-Left2Right-v1', 'FetchSlideOOD-Right2Right-v1', 'FetchSlideOOD-Right2Left-v1']
    elif 'Stack' in env:
        if 'far' in data_type:
            envs = ['FetchStackOOD-Near2Near-v1', 'FetchStackOOD-Near2Far-v1', 'FetchStackOOD-Far2Near-v1', 'FetchStackOOD-Far2Far-v1']
        else:
            envs = ['FetchStackOOD-Right2Left-v1', 'FetchStackOOD-Right2Right-v1', 'FetchStackOOD-Left2Left-v1', 'FetchStackOOD-Left2Right-v1']
    elif 'Block' in env:
        envs = ['HandBlockOOD-P2P-v0', 'HandBlockOOD-P2N-v0', 'HandBlockOOD-N2P-v0', 'HandBlockOOD-N2N-v0']
    else:
        envs = ['FetchPush-v1']
    return envs

def get_full_envname(name):
    dic = {
        'PointReach': 'Point2DLargeEnv-v1',
        'FetchReach':'FetchReach-v1',
        'FetchPush': 'FetchPush-v1',
        'FetchSlide': 'FetchSlide-v1',
        'FetchPick': 'FetchPickAndPlace-v1',
        'HandReach':'HandReach-v0'
    }
    if name in dic.keys():
        return dic[name]
    else:
        return name


def create_stats_ordered_dict(
        name,
        data,
        stat_prefix=None,
        always_show_all_stats=True,
        exclude_max_min=False,
):
    if stat_prefix is not None:
        name = "{} {}".format(stat_prefix, name)
    if isinstance(data, Number):
        return OrderedDict({name: data})

    if len(data) == 0:
        return OrderedDict()

    if isinstance(data, tuple):
        ordered_dict = OrderedDict()
        for number, d in enumerate(data):
            sub_dict = create_stats_ordered_dict(
                "{0}_{1}".format(name, number),
                d,
            )
            ordered_dict.update(sub_dict)
        return ordered_dict

    if isinstance(data, list):
        try:
            iter(data[0])
        except TypeError:
            pass
        else:
            data = np.concatenate(data)

    if (isinstance(data, np.ndarray) and data.size == 1
            and not always_show_all_stats):
        return OrderedDict({name: float(data)})

    stats = OrderedDict([
        (name + ' Mean', np.mean(data)),
        (name + ' Std', np.std(data)),
    ])
    if not exclude_max_min:
        stats[name + ' Max'] = np.max(data)
        stats[name + ' Min'] = np.min(data)
    return stats


def get_generic_path_information(paths, stat_prefix=''):
    """
    Get an OrderedDict with a bunch of statistic names and values.
    """
    statistics = OrderedDict()
    returns = [sum(path["rewards"]) for path in paths]

    rewards = np.vstack([path["rewards"] for path in paths])
    statistics.update(create_stats_ordered_dict('Rewards', rewards,
                                                stat_prefix=stat_prefix))
    statistics.update(create_stats_ordered_dict('Returns', returns,
                                                stat_prefix=stat_prefix))
    actions = [path["actions"] for path in paths]
    if len(actions[0].shape) == 1:
        actions = np.hstack([path["actions"] for path in paths])
    else:
        actions = np.vstack([path["actions"] for path in paths])
    statistics.update(create_stats_ordered_dict(
        'Actions', actions, stat_prefix=stat_prefix
    ))
    statistics['Num Paths'] = len(paths)

    return statistics


def get_average_returns(paths):
    returns = [sum(path["rewards"]) for path in paths]
    return np.mean(returns)


def get_path_lengths(paths):
    return [len(path['observations']) for path in paths]


def get_stat_in_paths(paths, dict_name, scalar_name):
    if len(paths) == 0:
        return np.array([[]])

    if type(paths[0][dict_name]) == dict:
        # Support rllab interface
        return [path[dict_name][scalar_name] for path in paths]

    return [
        [info[scalar_name] for info in path[dict_name]]
        for path in paths
    ]


def get_asset_full_path(file_name):
    return os.path.join(ENV_ASSET_DIR, file_name)

def concatenate_box_spaces(*spaces):
    """
    Assumes dtypes of all spaces are the of the same type
    """
    low = np.concatenate([space.low for space in spaces])
    high = np.concatenate([space.high for space in spaces])
    return Box(low=low, high=high, dtype=np.float32)
