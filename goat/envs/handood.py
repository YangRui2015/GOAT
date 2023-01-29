import os
import numpy as np

from gym import utils, error
from gym.envs.robotics import rotations, hand_env
from gym.envs.robotics.utils import robot_get_obs

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))


def quat_from_angle_and_axis(angle, axis):
    assert axis.shape == (3,)
    axis /= np.linalg.norm(axis)
    quat = np.concatenate([[np.cos(angle / 2.)], np.sin(angle / 2.) * axis])
    quat /= np.linalg.norm(quat)
    return quat


# Ensure we get the path separator correct on windows
MANIPULATE_BLOCK_XML = os.path.join('hand', 'manipulate_block.xml')


class ManipulateEnv(hand_env.HandEnv):
    def __init__(
        self, model_path, init_rotation, target_position, target_rotation, 
        target_position_range, reward_type, initial_qpos=None,
        randomize_initial_position=True, randomize_initial_rotation=True,
        distance_threshold=0.01, rotation_threshold=0.1, n_substeps=20, relative_control=False,
        ignore_z_target_rotation=False, 
    ):
        """Initializes a new Hand manipulation environment.

        Args:
            model_path (string): path to the environments XML file
            target_position (string): the type of target position:
                - ignore: target position is fully ignored, i.e. the object can be positioned arbitrarily
                - fixed: target position is set to the initial position of the object
                - random: target position is fully randomized according to target_position_range
            target_rotation (string): the type of target rotation:
                - ignore: target rotation is fully ignored, i.e. the object can be rotated arbitrarily
                - fixed: target rotation is set to the initial rotation of the object
                - xyz: fully randomized target rotation around the X, Y and Z axis
                - z: fully randomized target rotation around the Z axis
                - parallel: fully randomized target rotation around Z and axis-aligned rotation around X, Y
            ignore_z_target_rotation (boolean): whether or not the Z axis of the target rotation is ignored
            target_position_range (np.array of shape (3, 2)): range of the target_position randomization
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            randomize_initial_position (boolean): whether or not to randomize the initial position of the object
            randomize_initial_rotation (boolean): whether or not to randomize the initial rotation of the object
            distance_threshold (float, in meters): the threshold after which the position of a goal is considered achieved
            rotation_threshold (float, in radians): the threshold after which the rotation of a goal is considered achieved
            n_substeps (int): number of substeps the simulation runs on every call to step
            relative_control (boolean): whether or not the hand is actuated in absolute joint positions or relative to the current state
        """

        self.target_position = target_position
        self.target_rotation = target_rotation
        self.target_position_range = target_position_range
        self.parallel_quats = [rotations.euler2quat(r) for r in rotations.get_parallel_rotations()]
        self.randomize_initial_rotation = randomize_initial_rotation
        self.randomize_initial_position = randomize_initial_position
        self.distance_threshold = distance_threshold
        self.rotation_threshold = rotation_threshold
        self.reward_type = reward_type
        self.ignore_z_target_rotation = ignore_z_target_rotation
        self.init_rotation = init_rotation

        assert self.target_position in ['ignore', 'fixed', 'random']
        assert self.target_rotation in ['ignore', 'fixed', 'xyz', 'z', 'parallel', 'xyz-firsthalf', 'xyz-secondhalf', 'xy-ood']
        initial_qpos = initial_qpos or {}

        hand_env.HandEnv.__init__(
            self, model_path, n_substeps=n_substeps, initial_qpos=initial_qpos,
            relative_control=relative_control)


    def _get_achieved_goal(self):
        # Object position and rotation.
        object_qpos = self.sim.data.get_joint_qpos('object:joint')
        assert object_qpos.shape == (7,)
        return object_qpos

    def _goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        assert goal_a.shape[-1] == 7

        d_pos = np.zeros_like(goal_a[..., 0])
        d_rot = np.zeros_like(goal_b[..., 0])
        if self.target_position != 'ignore':
            delta_pos = goal_a[..., :3] - goal_b[..., :3]
            d_pos = np.linalg.norm(delta_pos, axis=-1)

        if self.target_rotation != 'ignore':
            quat_a, quat_b = goal_a[..., 3:], goal_b[..., 3:]

            if self.ignore_z_target_rotation:
                # Special case: We want to ignore the Z component of the rotation.
                # This code here assumes Euler angles with xyz convention. We first transform
                # to euler, then set the Z component to be equal between the two, and finally
                # transform back into quaternions.
                euler_a = rotations.quat2euler(quat_a)
                euler_b = rotations.quat2euler(quat_b)
                euler_a[2] = euler_b[2]
                quat_a = rotations.euler2quat(euler_a)

            # Subtract quaternions and extract angle between them.
            quat_diff = rotations.quat_mul(quat_a, rotations.quat_conjugate(quat_b))
            angle_diff = 2 * np.arccos(np.clip(quat_diff[..., 0], -1., 1.))
            d_rot = angle_diff
        assert d_pos.shape == d_rot.shape
        return d_pos, d_rot

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        if self.reward_type == 'sparse':
            success = self._is_success(achieved_goal, goal).astype(np.float32)
            return (success - 1.)
        else:
            d_pos, d_rot = self._goal_distance(achieved_goal, goal)
            # We weigh the difference in position to avoid that `d_pos` (in meters) is completely
            # dominated by `d_rot` (in radians).
            return -(10. * d_pos + d_rot)

    # RobotEnv methods
    # ----------------------------

    def _is_success(self, achieved_goal, desired_goal):
        d_pos, d_rot = self._goal_distance(achieved_goal, desired_goal)
        achieved_pos = (d_pos < self.distance_threshold).astype(np.float32)
        achieved_rot = (d_rot < self.rotation_threshold).astype(np.float32)
        achieved_both = achieved_pos * achieved_rot
        return achieved_both

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.sim.forward()

        initial_qpos = self.sim.data.get_joint_qpos('object:joint').copy()
        initial_pos, initial_quat = initial_qpos[:3], initial_qpos[3:]
        assert initial_qpos.shape == (7,)
        assert initial_pos.shape == (3,)
        assert initial_quat.shape == (4,)
        initial_qpos = None

        # Randomization initial rotation.
        if self.randomize_initial_rotation:
            if self.init_rotation == 'xyz-firsthalf':
                # angle -pi~0, axis -1~0 complements angle 0~pi, axis 0~1
                angle = self.np_random.uniform(-np.pi, 0)
                axis = self.np_random.uniform(-1., 1., size=3)
                axis[2] = self.np_random.uniform(0., 1.) 
                offset_quat = quat_from_angle_and_axis(angle, axis)
                initial_quat = rotations.quat_mul(initial_quat, offset_quat) 
                # this equals offset_quat because initial_quat is 1,0,0,0.
            elif self.init_rotation == 'xyz-secondhalf':
                angle = self.np_random.uniform(0, np.pi)
                axis = self.np_random.uniform(-1., 1., size=3)
                axis[2] = self.np_random.uniform(0., 1.) 
                offset_quat = quat_from_angle_and_axis(angle, axis)
                initial_quat = rotations.quat_mul(initial_quat, offset_quat)
            elif self.init_rotation == 'z':
                angle = self.np_random.uniform(-np.pi, np.pi)
                axis = np.array([0., 0., 1.])
                offset_quat = quat_from_angle_and_axis(angle, axis)
                initial_quat = rotations.quat_mul(initial_quat, offset_quat)
            elif self.init_rotation == 'xy':
                angle = self.np_random.uniform(-np.pi, np.pi)
                axis = self.np_random.uniform(-1., 1., size=3)
                axis[2] = 0
                offset_quat = quat_from_angle_and_axis(angle, axis)
                initial_quat = rotations.quat_mul(initial_quat, offset_quat)
            elif self.target_rotation == 'parallel':
                angle = self.np_random.uniform(-np.pi, np.pi)
                axis = np.array([0., 0., 1.])
                z_quat = quat_from_angle_and_axis(angle, axis)
                parallel_quat = self.parallel_quats[self.np_random.randint(len(self.parallel_quats))]
                offset_quat = rotations.quat_mul(z_quat, parallel_quat)
                initial_quat = rotations.quat_mul(initial_quat, offset_quat)
            elif self.target_rotation in ['xyz', 'ignore']:
                angle = self.np_random.uniform(-np.pi, np.pi)
                axis = self.np_random.uniform(-1., 1., size=3)
                offset_quat = quat_from_angle_and_axis(angle, axis)
                initial_quat = rotations.quat_mul(initial_quat, offset_quat)
            elif self.target_rotation == 'fixed':
                pass
            else:
                raise error.Error('Unknown target_rotation option "{}".'.format(self.target_rotation))

        # Randomize initial position.
        if self.randomize_initial_position:
            if self.target_position != 'fixed':
                initial_pos += self.np_random.normal(size=3, scale=0.005)

        initial_quat /= np.linalg.norm(initial_quat)
        initial_qpos = np.concatenate([initial_pos, initial_quat])
        self.sim.data.set_joint_qpos('object:joint', initial_qpos)

        def is_on_palm():
            self.sim.forward()
            cube_middle_idx = self.sim.model.site_name2id('object:center')
            cube_middle_pos = self.sim.data.site_xpos[cube_middle_idx]
            is_on_palm = (cube_middle_pos[2] > 0.04)
            return is_on_palm

        # Run the simulation for a bunch of timesteps to let everything settle in.
        for _ in range(10):
            self._set_action(np.zeros(20))
            try:
                self.sim.step()
            except mujoco_py.MujocoException:
                return False
        return is_on_palm()

    def _sample_goal(self):
        # Select a goal for the object position.
        target_pos = None
        if self.target_position == 'random':
            assert self.target_position_range.shape == (3, 2)
            offset = self.np_random.uniform(self.target_position_range[:, 0], self.target_position_range[:, 1])
            assert offset.shape == (3,)
            target_pos = self.sim.data.get_joint_qpos('object:joint')[:3] + offset
        elif self.target_position in ['ignore', 'fixed']:
            target_pos = self.sim.data.get_joint_qpos('object:joint')[:3]
        else:
            raise error.Error('Unknown target_position option "{}".'.format(self.target_position))
        assert target_pos is not None
        assert target_pos.shape == (3,)

        # Select a goal for the object rotation.
        target_quat = None
        if self.target_rotation == 'z':
            angle = self.np_random.uniform(-np.pi, np.pi)
            axis = np.array([0., 0., 1.])
            target_quat = quat_from_angle_and_axis(angle, axis)
        elif self.target_rotation == 'parallel':
            angle = self.np_random.uniform(-np.pi, np.pi)
            axis = np.array([0., 0., 1.])
            target_quat = quat_from_angle_and_axis(angle, axis)
            parallel_quat = self.parallel_quats[self.np_random.randint(len(self.parallel_quats))]
            target_quat = rotations.quat_mul(target_quat, parallel_quat)
        elif self.target_rotation == 'xyz':
            angle = self.np_random.uniform(-np.pi, np.pi)
            axis = self.np_random.uniform(-1., 1., size=3)
            target_quat = quat_from_angle_and_axis(angle, axis)
        elif self.target_rotation in ['ignore', 'fixed']:
            target_quat = self.sim.data.get_joint_qpos('object:joint')
        elif self.target_rotation == 'xyz-firsthalf':
            # angle -pi~0, axis -1~0 complements angle 0~pi, axis 0~1
            angle = self.np_random.uniform(-np.pi, 0)
            axis = self.np_random.uniform(-1., 1., size=3)
            axis[2] = self.np_random.uniform(0., 1.) 
            target_quat = quat_from_angle_and_axis(angle, axis)
        elif self.target_rotation == 'xyz-secondhalf':
            angle = self.np_random.uniform(0, np.pi)
            axis = self.np_random.uniform(-1., 1., size=3)
            axis[2] = self.np_random.uniform(0., 1.) 
            target_quat = quat_from_angle_and_axis(angle, axis)
        elif self.target_rotation == 'xy-ood':
            angle = self.np_random.uniform(-np.pi, np.pi)
            axis = self.np_random.uniform(-1., 1., size=3)
            axis[2] = 0
            target_quat = quat_from_angle_and_axis(angle, axis)
        elif self.target_rotation == 'z-ood':
            angle = self.np_random.uniform(-np.pi, np.pi)
            axis = np.array([0., 0., 1.])
            target_quat = quat_from_angle_and_axis(angle, axis)
        else:
            raise error.Error('Unknown target_rotation option "{}".'.format(self.target_rotation))
        assert target_quat is not None
        assert target_quat.shape == (4,)

        target_quat /= np.linalg.norm(target_quat)  # normalized quaternion
        goal = np.concatenate([target_pos, target_quat])
        return goal

    def _render_callback(self):
        # Assign current state to target object but offset a bit so that the actual object
        # is not obscured.
        goal = self.goal.copy()
        assert goal.shape == (7,)
        if self.target_position == 'ignore':
            # Move the object to the side since we do not care about it's position.
            goal[0] += 0.15
        self.sim.data.set_joint_qpos('target:joint', goal)
        self.sim.data.set_joint_qvel('target:joint', np.zeros(6))

        if 'object_hidden' in self.sim.model.geom_names:
            hidden_id = self.sim.model.geom_name2id('object_hidden')
            self.sim.model.geom_rgba[hidden_id, 3] = 1.
        self.sim.forward()

    def _get_obs(self):
        robot_qpos, robot_qvel = robot_get_obs(self.sim)
        object_qvel = self.sim.data.get_joint_qvel('object:joint')
        achieved_goal = self._get_achieved_goal().ravel()  # this contains the object position + rotation
        observation = np.concatenate([robot_qpos, robot_qvel, object_qvel, achieved_goal])
        return {
            'observation': observation.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.ravel().copy(),
        }

class HandBlockOODEnv(ManipulateEnv, utils.EzPickle):
    def __init__(self, init_rotation=None, target_position='random', target_rotation='xyz', reward_type='sparse'):
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        ManipulateEnv.__init__(self,
            model_path=MANIPULATE_BLOCK_XML,
            init_rotation=init_rotation, target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type)


FINGERTIP_SITE_NAMES = [
    'robot0:S_fftip',
    'robot0:S_mftip',
    'robot0:S_rftip',
    'robot0:S_lftip',
    'robot0:S_thtip',
]


DEFAULT_INITIAL_QPOS = {
    'robot0:WRJ1': -0.16514339750464327,
    'robot0:WRJ0': -0.31973286565062153,
    'robot0:FFJ3': 0.14340512546557435,
    'robot0:FFJ2': 0.32028208333591573,
    'robot0:FFJ1': 0.7126053607727917,
    'robot0:FFJ0': 0.6705281001412586,
    'robot0:MFJ3': 0.000246444303701037,
    'robot0:MFJ2': 0.3152655251085491,
    'robot0:MFJ1': 0.7659800313729842,
    'robot0:MFJ0': 0.7323156897425923,
    'robot0:RFJ3': 0.00038520700007378114,
    'robot0:RFJ2': 0.36743546201985233,
    'robot0:RFJ1': 0.7119514095008576,
    'robot0:RFJ0': 0.6699446327514138,
    'robot0:LFJ4': 0.0525442258033891,
    'robot0:LFJ3': -0.13615534724474673,
    'robot0:LFJ2': 0.39872030433433003,
    'robot0:LFJ1': 0.7415570009679252,
    'robot0:LFJ0': 0.704096378652974,
    'robot0:THJ4': 0.003673823825070126,
    'robot0:THJ3': 0.5506291436028695,
    'robot0:THJ2': -0.014515151997119306,
    'robot0:THJ1': -0.0015229223564485414,
    'robot0:THJ0': -0.7894883021600622,
}

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('hand', 'reach.xml')

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class HandReachOODEnv(hand_env.HandEnv, utils.EzPickle):
    def __init__(
        self, distance_threshold=0.01, n_substeps=20, relative_control=False,
        initial_qpos=DEFAULT_INITIAL_QPOS, reward_type='sparse', 
        ood_g_scale=None, in_g_scale=0.,
    ):
        utils.EzPickle.__init__(**locals())
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.ood_g_scale = ood_g_scale
        self.in_g_scale = in_g_scale

        hand_env.HandEnv.__init__(
            self, MODEL_XML_PATH, n_substeps=n_substeps, initial_qpos=initial_qpos,
            relative_control=relative_control)

    def _get_achieved_goal(self):
        goal = [self.sim.data.get_site_xpos(name) for name in FINGERTIP_SITE_NAMES]
        return np.array(goal).flatten()

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    # RobotEnv methods
    # ----------------------------
    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        self.sim.forward()

        self.initial_goal = self._get_achieved_goal().copy()
        self.palm_xpos = self.sim.data.body_xpos[self.sim.model.body_name2id('robot0:palm')].copy()

    def _get_obs(self):
        robot_qpos, robot_qvel = robot_get_obs(self.sim)
        achieved_goal = self._get_achieved_goal().ravel()
        observation = np.concatenate([robot_qpos, robot_qvel, achieved_goal])
        return {
            'observation': observation.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _sample_goal(self):
        thumb_name = 'robot0:S_thtip'
        finger_names = [name for name in FINGERTIP_SITE_NAMES if name != thumb_name]
        finger_name = self.np_random.choice(finger_names)

        thumb_idx = FINGERTIP_SITE_NAMES.index(thumb_name)
        finger_idx = FINGERTIP_SITE_NAMES.index(finger_name)
        assert thumb_idx != finger_idx

        # Pick a meeting point above the hand.
        meeting_pos = self.palm_xpos + np.array([0.0, -0.09, 0.05])
        if self.ood_g_scale is None:
            meeting_pos += self.np_random.normal(scale=0.005, size=meeting_pos.shape)
        else:
            shift = self.np_random.uniform(self.in_g_scale, self.ood_g_scale, size=meeting_pos.shape)
            random_neg = self.np_random.uniform(-1, 1, size=meeting_pos.shape)
            random_neg[random_neg < 0] = -1
            random_neg[random_neg >= 0] = 1
            shift = shift * random_neg
            meeting_pos += shift

        # Slightly move meeting goal towards the respective finger to avoid that they
        # overlap.
        goal = self.initial_goal.copy().reshape(-1, 3)
        for idx in [thumb_idx, finger_idx]:
            offset_direction = (meeting_pos - goal[idx])
            offset_direction /= np.linalg.norm(offset_direction)
            goal[idx] = meeting_pos - 0.005 * offset_direction

        if self.np_random.uniform() < 0.1 and (self.in_g_scale is None or (self.in_g_scale is not None and self.in_g_scale == 0)):
            # With some probability, ask all fingers to move back to the origin.
            # This avoids that the thumb constantly stays near the goal position already.
            goal = self.initial_goal.copy()
        return goal.flatten()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _render_callback(self):
        # Visualize targets.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        goal = self.goal.reshape(5, 3)
        for finger_idx in range(5):
            site_name = 'target{}'.format(finger_idx)
            site_id = self.sim.model.site_name2id(site_name)
            self.sim.model.site_pos[site_id] = goal[finger_idx] - sites_offset[site_id]

        # Visualize finger positions.
        achieved_goal = self._get_achieved_goal().reshape(5, 3)
        for finger_idx in range(5):
            site_name = 'finger{}'.format(finger_idx)
            site_id = self.sim.model.site_name2id(site_name)
            self.sim.model.site_pos[site_id] = achieved_goal[finger_idx] - sites_offset[site_id]
        self.sim.forward()




