import gym
from gym.envs.mujoco import mujoco_env
from gym import utils
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def linear_schedule(start_sigma: float, end_sigma: float, duration: int, t: int):
    return end_sigma + (1 - min(t / duration, 1)) * (start_sigma - end_sigma)

# saving frames 

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):
    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

# MBPO environments

MBPO_ENVIRONMENT_SPECS = (
	{
        'id': 'AntTruncatedObs-v2',
        'entry_point': (f'utils.env:AntTruncatedObsEnv'),
        'max_episode_steps': 1000,
    },
	{
        'id': 'HumanoidTruncatedObs-v2',
        'entry_point': (f'utils.env:HumanoidTruncatedObsEnv'),
        'max_episode_steps': 1000,
    },
)

def _register_environments(register, specs):
    for env in specs:
        register(**env)

    gym_ids = tuple(environment_spec['id'] for environment_spec in specs)
    return gym_ids

def register_mbpo_environments():
    _register_environments(gym.register, MBPO_ENVIRONMENT_SPECS)

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

class HumanoidTruncatedObsEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
        COM inertia (cinert), COM velocity (cvel), actuator forces (qfrc_actuator), 
        and external forces (cfrc_ext) are removed from the observation.
        Otherwise identical to Humanoid-v2 from
        https://github.com/openai/gym/blob/master/gym/envs/mujoco/humanoid.py
    """
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'humanoid.xml', 5)
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               # data.cinert.flat,
                               # data.cvel.flat,
                               # data.qfrc_actuator.flat,
                               # data.cfrc_ext.flat
                               ])

    def step(self, a):
        pos_before = mass_center(self.model, self.sim)
        self.do_simulation(a, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)
        alive_bonus = 5.0
        data = self.sim.data
        lin_vel_cost = 0.25 * (pos_after - pos_before) / self.model.opt.timestep
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        return self._get_obs(), reward, done, dict(reward_linvel=lin_vel_cost, reward_quadctrl=-quad_ctrl_cost, reward_alive=alive_bonus, reward_impact=-quad_impact_cost)

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20

class AntTruncatedObsEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
        External forces (sim.data.cfrc_ext) are removed from the observation.
        Otherwise identical to Ant-v2 from
        https://github.com/openai/gym/blob/master/gym/envs/mujoco/ant.py
    """
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'ant.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            # np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

class GymActionRepeatWrapper(gym.Wrapper):
	def __init__(self, env, num_repeats):
		assert '-v4' in env.unwrapped.spec.id
		super().__init__(env)
		self._env = env
		self._num_repeats = num_repeats

	def step(self, action):
		reward = 0.0
		notdone = True
		for i in range(self._num_repeats):
			state, rew, done, info = self._env.step(action)
			notdone = not done
			reward += (rew) * (notdone)
			notdone *= notdone
			if done:
				break
		return state, reward, done,  info