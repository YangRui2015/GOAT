import gym
from gym.envs.registration import register

def register_envs():
    register(
        id='PointFixedEnv-v1',
        entry_point='goat.envs.point2d:Point2DEnv',
        tags={
            'git-commit-hash': '4efe2be',
            'author': 'Vitchyr'
        },
        kwargs={
            'action_scale': 1,
            'images_are_rgb': True,
            'target_radius': 1,
            'ball_radius': 0.5,
            "circle_radius": 10,
            'boundary_dist': 11,
            'render_onscreen': False,
            'show_goal': True,
            'render_size':512,
            'get_image_base_render_size': (48, 48),
            'bg_color': 'white',
            'fixed_goal_set':True,
            'fixed_goal_set_id': 0,
            'fixed_init_position': (0,0),
            'randomize_position_on_reset': False
        },
    )
    register(
        id='PointFixedLargeEnv-v1',
        entry_point='goat.envs.point2d:Point2DEnv',
        tags={
            'git-commit-hash': '4efe2be',
            'author': 'Vitchyr'
        },
        kwargs={
            'action_scale': 1,
            'images_are_rgb': True,
            'target_radius': 1,
            'ball_radius': 0.5,
            "circle_radius": 20,
            'boundary_dist': 21,
            'render_onscreen': False,
            'show_goal': True,
            'render_size':512,
            'get_image_base_render_size': (48, 48),
            'bg_color': 'white',
            'fixed_goal_set':True,
            'fixed_goal_set_id': 0,
            'fixed_init_position': (0,0),
            'randomize_position_on_reset': False
        },
    )

    # register OOD envs for evaluation
    register(
        id='FetchReachOOD-Near-v1',
        entry_point='goat.envs.fetch_ood:FetchReachOODEnv',
         kwargs={
            'goal_type': 'circle' , 
            'ood_g_range': [0.,0.15], 
            'target_range': 0.15 
         },
    )
    register(
        id='FetchReachOOD-Far-v1',
        entry_point='goat.envs.fetch_ood:FetchReachOODEnv',
         kwargs={
            'goal_type': 'circle' , 
            'ood_g_range': [0.15, 0.3],
            'target_range': 0.3
         },
    )
    register(
        id='FetchReachOOD-Left-v1',
        entry_point='goat.envs.fetch_ood:FetchReachOODEnv',
         kwargs={
            'goal_type': 'left' , 
            'target_range': 0.15 
         },
    )
    register(
        id='FetchReachOOD-Right-v1',
        entry_point='goat.envs.fetch_ood:FetchReachOODEnv',
         kwargs={
            'goal_type': 'right' , 
            'target_range': 0.15
         },
    )
    # Push
    register(
        id='FetchPushOOD-Near2Near-v1',
        entry_point='goat.envs.fetch_ood:FetchPushOODEnv',
         kwargs={
            'goal_type': 'circle' , 
            'ood_g_range': [0.,0.15], 
            'obj_range': 0.15, 
            'target_range': 0.15 
         },
    )
    register(
        id='FetchPushOOD-Near2Far-v1',
        entry_point='goat.envs.fetch_ood:FetchPushOODEnv',
         kwargs={
            'goal_type': 'circle' , 
            'ood_g_range': [0.15, 0.3],
            'obj_range': 0.15, 
            'target_range': 0.3
         },
    )
    register(
        id='FetchPushOOD-Far2Near-v1',
        entry_point='goat.envs.fetch_ood:FetchPushOODEnv',
         kwargs={
            'goal_type': 'circle' , 
            'ood_g_range': [0., 0.15],
            'ood_obj_range': [0.15, 0.3],
            'obj_range': 0.3, 
            'target_range': 0.15
         },
    )
    register(
        id='FetchPushOOD-Far2Far-v1',
        entry_point='goat.envs.fetch_ood:FetchPushOODEnv',
         kwargs={
            'goal_type': 'circle' , 
            'ood_g_range': [0.15, 0.3],
            'ood_obj_range': [0.15, 0.3],
            'obj_range': 0.3, 
            'target_range': 0.3
         },
    )
    register(
        id='FetchPushOOD-Right2Left-v1',
        entry_point='goat.envs.fetch_ood:FetchPushOODEnv',
         kwargs={
            'goal_type': 'left',
            'initial_type': 'right',
            'obj_range': 0.15,
            'target_range': 0.15 
         },
    )
    register(
        id='FetchPushOOD-Left2Right-v1',
        entry_point='goat.envs.fetch_ood:FetchPushOODEnv',
         kwargs={
            'goal_type': 'right',
            'initial_type': 'left',
            'obj_range': 0.15,
            'target_range': 0.15 
         },
    )
    register(
        id='FetchPushOOD-Left2Left-v1',
        entry_point='goat.envs.fetch_ood:FetchPushOODEnv',
         kwargs={
            'goal_type': 'left',
            'initial_type': 'left',
            'obj_range': 0.15,
            'target_range': 0.15 
         },
    )
    register(
        id='FetchPushOOD-Right2Right-v1',
        entry_point='goat.envs.fetch_ood:FetchPushOODEnv',
         kwargs={
            'goal_type': 'right',
            'initial_type': 'right',
            'obj_range': 0.15,
            'target_range': 0.15 
         },
    )
    ## Pick
    register(
        id='FetchPickOOD-Right2Right-v1',
        entry_point='goat.envs.fetch_ood:FetchPickOODEnv',
         kwargs={
            'goal_type': 'right',
            'initial_type': 'right',
            'obj_range': 0.15,
            'target_range': 0.15 
         },
    )
    register(
        id='FetchPickOOD-Right2Left-v1',
        entry_point='goat.envs.fetch_ood:FetchPickOODEnv',
         kwargs={
            'goal_type': 'left',
            'initial_type': 'right',
            'obj_range': 0.15,
            'target_range': 0.15 
         },
    )
    register(
        id='FetchPickOOD-Left2Left-v1',
        entry_point='goat.envs.fetch_ood:FetchPickOODEnv',
         kwargs={
            'goal_type': 'left',
            'initial_type': 'left',
            'obj_range': 0.15,
            'target_range': 0.15 
         },
    )
    register(
        id='FetchPickOOD-Left2Right-v1',
        entry_point='goat.envs.fetch_ood:FetchPickOODEnv',
         kwargs={
            'goal_type': 'right',
            'initial_type': 'left',
            'obj_range': 0.15,
            'target_range': 0.15 
         },
    )
    register(
        id='FetchPickOOD-Low2High-v1',
        entry_point='goat.envs.fetch_ood:FetchPickOODEnv',
         kwargs={
            'goal_type': 'height' , 
            'ood_g_range': [0.6, 0.9],
            'obj_range': 0.15, 
            'target_range': 0.15
         },
    )
    register(
        id='FetchPickOOD-Low2Low-v1',
        entry_point='goat.envs.fetch_ood:FetchPickOODEnv',
         kwargs={
            'goal_type': 'height' , 
            'ood_g_range': [0, 0.6],
            'obj_range': 0.15, 
            'target_range': 0.15
         },
    )
    # Slide
    register(
        id='FetchSlideOOD-Left2Left-v1',
        entry_point='goat.envs.fetch_ood:FetchSlideOODEnv',
         kwargs={
            'goal_type': 'left',
            'initial_type': 'left',
            'ood_g_range': [-0.3, 0]
         },
    )
    register(
        id='FetchSlideOOD-Left2Right-v1',
        entry_point='goat.envs.fetch_ood:FetchSlideOODEnv',
         kwargs={
            'goal_type': 'right',
            'initial_type': 'left',
            'ood_g_range': [-0.3, 0]
         },
    )
    register(
        id='FetchSlideOOD-Right2Right-v1',
        entry_point='goat.envs.fetch_ood:FetchSlideOODEnv',
         kwargs={
            'goal_type': 'right',
            'initial_type': 'right',
            'ood_g_range': [-0.3, 0]
         },
    )
    register(
        id='FetchSlideOOD-Right2Left-v1',
        entry_point='goat.envs.fetch_ood:FetchSlideOODEnv',
         kwargs={
            'goal_type': 'left',
            'initial_type': 'right',
            'ood_g_range': [-0.3, 0]
         },
    )
    register(
        id='FetchSlideOOD-Near2Near-v1',
        entry_point='goat.envs.fetch_ood:FetchSlideOODEnv',
         kwargs={
            'goal_type': 'distance',
            'ood_g_range': [-0.3, 0], # abs pos [1.1, 1.4]
         },
    )
    register(
        id='FetchSlideOOD-Near2Far-v1',
        entry_point='goat.envs.fetch_ood:FetchSlideOODEnv',
         kwargs={
            'goal_type': 'distance',
            'ood_g_range': [0, 0.2], # abs pos [1.4, 1.6]
         },
    )
   
    # Hand Reach
    register(
        id='HandReachOOD-Near-v0',
        entry_point='goat.envs.handood:HandReachOODEnv',
         kwargs={
            'ood_g_scale': 0.005,
            'in_g_scale': 0.,
         },
    )
    register(
        id='HandReachOOD-Far-v0',
        entry_point='goat.envs.handood:HandReachOODEnv',
         kwargs={
            'ood_g_scale': 0.015,
            'in_g_scale': 0.005,
         },
    )

