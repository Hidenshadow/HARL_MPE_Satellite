import copy

import numpy as np
from gymnasium.utils import EzPickle
from pettingzoo.utils.conversions import parallel_wrapper_fn

from .._mpe_utils.core import Agent, Landmark, World
from .._mpe_utils.scenario import BaseScenario
from .._mpe_utils.simple_env import SimpleEnv, make_env

DEFAULT_MAP_NAME = "2x2-4A"
GLOBAL_BOUNDS = [-1.0, 1.0, -1.0, 1.0]


def _agent(fov_angle, radius):
    return {"fov_angle": float(fov_angle), "radius": float(radius)}


def _agent_groups(num_agents, groups):
    agent_configs = [None] * num_agents
    for indices, fov_angle, radius in groups:
        for index in indices:
            agent_configs[index] = _agent(fov_angle, radius)

    missing = [index for index, config in enumerate(agent_configs) if config is None]
    if missing:
        raise ValueError(f"Missing satellite agent configs for agents {missing}.")
    return agent_configs


def _config(size, map_name, targets, grid_size, episode_length, groups):
    num_agents = grid_size[0] * grid_size[1]
    return {
        "size": size,
        "map_name": map_name,
        "targets": int(targets),
        "grid_size": tuple(grid_size),
        "num_agents": num_agents,
        "episode_length": int(episode_length),
        "global_bounds": list(GLOBAL_BOUNDS),
        "agent_configs": _agent_groups(num_agents, groups),
    }


SATELLITE_MAP_CONFIGS = {
    "2x2-4A": _config(
        "Small",
        "2x2-4A",
        targets=40,
        grid_size=(2, 2),
        episode_length=100,
        groups=[(range(4), 15, 1.25)],
    ),
    "2x2-2A2C": _config(
        "Small",
        "2x2-2A2C",
        targets=40,
        grid_size=(2, 2),
        episode_length=100,
        groups=[([0, 3], 15, 1.25), ([1, 2], 10, 1.5)],
    ),
    "3x2-6A": _config(
        "Small",
        "3x2-6A",
        targets=60,
        grid_size=(2, 3),
        episode_length=125,
        groups=[(range(6), 15, 1.25)],
    ),
    "3x2-2A2B2C": _config(
        "Small",
        "3x2-2A2B2C",
        targets=60,
        grid_size=(2, 3),
        episode_length=125,
        groups=[([0, 5], 15, 1.25), ([1, 4], 10, 1.5), ([2, 3], 30, 1.0)],
    ),
    "4x2-8A": _config(
        "Medium",
        "4x2-8A",
        targets=80,
        grid_size=(2, 4),
        episode_length=150,
        groups=[(range(8), 15, 1.25)],
    ),
    "4x2-2A4B2C": _config(
        "Medium",
        "4x2-2A4B2C",
        targets=80,
        grid_size=(2, 4),
        episode_length=150,
        groups=[([0, 7], 15, 1.25), ([1, 6], 10, 1.5), (range(2, 6), 30, 1.0)],
    ),
    "3x3-9A": _config(
        "Medium",
        "3x3-9A",
        targets=90,
        grid_size=(3, 3),
        episode_length=150,
        groups=[(range(9), 15, 1.25)],
    ),
    "3x3-4A1B4C": _config(
        "Medium",
        "3x3-4A1B4C",
        targets=90,
        grid_size=(3, 3),
        episode_length=150,
        groups=[([1, 3, 5, 7], 15, 1.25), ([0, 2, 6, 8], 10, 1.5), ([4], 30, 1.0)],
    ),
    "4x3-12A": _config(
        "Large",
        "4x3-12A",
        targets=96,
        grid_size=(3, 4),
        episode_length=150,
        groups=[(range(12), 15, 1.25)],
    ),
    "4x3-6A2B4C": _config(
        "Large",
        "4x3-6A2B4C",
        targets=96,
        grid_size=(3, 4),
        episode_length=150,
        groups=[
            ([1, 3, 5, 6, 8, 10], 15, 1.25),
            ([0, 2, 9, 11], 10, 1.5),
            ([4, 7], 30, 1.0),
        ],
    ),
    "4x4-16A": _config(
        "Large",
        "4x4-16A",
        targets=112,
        grid_size=(4, 4),
        episode_length=150,
        groups=[(range(16), 15, 1.25)],
    ),
    "4x4-8A4B4C": _config(
        "Large",
        "4x4-8A4B4C",
        targets=112,
        grid_size=(4, 4),
        episode_length=150,
        groups=[
            ([1, 2, 4, 7, 8, 11, 13, 14], 15, 1.25),
            ([0, 3, 12, 15], 10, 1.5),
            ([5, 6, 9, 10], 30, 1.0),
        ],
    ),
}


def available_maps():
    return tuple(SATELLITE_MAP_CONFIGS.keys())


def get_satellite_config(map_name=DEFAULT_MAP_NAME):
    if map_name not in SATELLITE_MAP_CONFIGS:
        maps = ", ".join(available_maps())
        raise ValueError(f"Unknown satellite map '{map_name}'. Available maps: {maps}")
    return copy.deepcopy(SATELLITE_MAP_CONFIGS[map_name])


def _build_region_bounds(grid_size, global_bounds):
    rows, cols = grid_size
    x_min, x_max, y_min, y_max = global_bounds
    x_edges = np.linspace(x_min, x_max, cols + 1)
    y_edges = np.linspace(y_min, y_max, rows + 1)

    bounds = []
    for row in range(rows):
        for col in range(cols):
            bounds.append([
                float(x_edges[col]),
                float(x_edges[col + 1]),
                float(y_edges[row]),
                float(y_edges[row + 1]),
            ])
    return bounds


class raw_env(SimpleEnv, EzPickle):
    def __init__(
            self,
            N=None,
            local_ratio=0.5,
            max_cycles=None,
            continuous_actions=False,
            render_mode=None,
            map_name=DEFAULT_MAP_NAME,
    ):
        config = get_satellite_config(map_name)
        if N is not None and N != config["num_agents"]:
            raise ValueError(
                f"Map {map_name} has {config['num_agents']} agents, but N={N} was requested."
            )
        if max_cycles is None:
            max_cycles = config["episode_length"]

        EzPickle.__init__(self, N, local_ratio, max_cycles, continuous_actions, render_mode, map_name)
        assert 0.0 <= local_ratio <= 1.0, "local_ratio is a proportion. Must be between 0 and 1."

        scenario = Scenario(config)
        world = scenario.make_world()
        super().__init__(
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            local_ratio=local_ratio,
        )
        self.metadata["name"] = f"Satellite-{map_name}"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.map_name = config["map_name"]
        self.num_agents = config["num_agents"]
        self.num_landmarks = config["targets"]
        self.global_bounds = config["global_bounds"]
        self.agent_configs = config["agent_configs"]
        self.landmark_visits = {}
        self.rew = 0

        self.region_bounds = _build_region_bounds(
            config["grid_size"], self.global_bounds
        )

    def make_world(self):
        world = World()
        world.dim_c = 2
        num_agents = self.num_agents
        num_landmarks = self.num_landmarks
        world.collaborative = True
        world.num_agents = num_agents

        world.agents = [Agent() for _ in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f'agent {i}'
            agent.collide = False
            agent.silent = True
            agent.size = 0.05
            agent.max_view_distance = self.agent_configs[i]["radius"]
            agent.fov_angle = self.agent_configs[i]["fov_angle"]
            agent.view_angle_range = np.deg2rad(agent.fov_angle) / 2.0

        world.landmarks = [Landmark() for _ in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = f'landmark {i}'
            landmark.collide = False
            landmark.movable = False

        self.landmark_visits = {l.name: 0 for l in world.landmarks}
        return world

    def reset_world(self, world, np_random):
        self.rew = 0

        self.agent_colors = [np_random.random(3) for _ in range(len(world.agents))]

        for i, agent in enumerate(world.agents):
            x_min, x_max, y_min, y_max = self.region_bounds[i % len(self.region_bounds)]
            agent.state.p_pos = np_random.uniform([x_min, y_min], [x_max, y_max])
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

            agent.region_id = i
            agent.region_bounds = self.region_bounds[i % len(self.region_bounds)]
            agent.color = self.agent_colors[i]
            agent.state.theta = np_random.uniform(-np.pi, np.pi)

        num_landmarks = self.num_landmarks
        while len(world.landmarks) < num_landmarks:
            world.landmarks.append(Landmark())
        world.landmarks = world.landmarks[:num_landmarks]

        for landmark_index, landmark in enumerate(world.landmarks):
            region_id = landmark_index % len(self.region_bounds)
            x_min, x_max, y_min, y_max = self.region_bounds[region_id]
            landmark.name = f"landmark {landmark_index}"
            landmark.collide = False
            landmark.movable = False
            landmark.state.p_pos = np.array([
                np_random.uniform(x_min, x_max),
                np_random.uniform(y_min, y_max)
            ])
            landmark.color = np.array([0.25, 0.25, 0.25])
            landmark.state.p_vel = np_random.uniform(-0.02, 0.02, world.dim_p)
            landmark.region_id = region_id

        self.landmark_visits = {l.name: 0 for l in world.landmarks}

    def observation(self, agent, world):
        max_view_distance = agent.max_view_distance
        view_angle_range = agent.view_angle_range
        agent_theta = agent.state.theta

        x_min, x_max, y_min, y_max = agent.region_bounds

        obs_base = np.array([
            *agent.state.p_vel,
            *agent.state.p_pos,
            x_min, x_max, y_min, y_max
        ], dtype=np.float32)

        visible_agents = []
        visible_landmarks = []

        for other_agent in world.agents:
            if other_agent is agent:
                continue
            rel_pos = other_agent.state.p_pos - agent.state.p_pos
            dist = np.linalg.norm(rel_pos)
            if dist <= max_view_distance:
                agent_angle = np.arctan2(rel_pos[1], rel_pos[0])
                angle_diff = (agent_angle - agent_theta + np.pi) % (2 * np.pi) - np.pi
                if abs(angle_diff) <= view_angle_range:
                    relative_theta = (other_agent.state.theta - agent_theta + np.pi) % (2 * np.pi) - np.pi
                    visible_agents.append((dist, rel_pos, angle_diff, relative_theta))

        visible_agents.sort(key=lambda x: x[0])
        agent_obs = []
        for _, rel_pos, angle, relative_theta in visible_agents:
            agent_obs.extend([rel_pos[0], rel_pos[1], angle, relative_theta])
        max_agents = len(world.agents) - 1
        while len(agent_obs) < max_agents * 4:
            agent_obs.extend([0.0, 0.0, 0.0, 0.0])
        if len(agent_obs) > max_agents * 4:
            agent_obs = agent_obs[:max_agents * 4]

        for entity in world.landmarks:
            rel_pos = entity.state.p_pos - agent.state.p_pos
            dist = np.linalg.norm(rel_pos)
            if dist <= max_view_distance:
                landmark_angle = np.arctan2(rel_pos[1], rel_pos[0])
                angle_diff = (landmark_angle - agent_theta + np.pi) % (2 * np.pi) - np.pi
                if abs(angle_diff) <= view_angle_range:
                    visible_landmarks.append((dist, rel_pos, angle_diff))

        landmark_obs = []
        for _, rel_pos, angle in visible_landmarks:
            landmark_obs.extend([rel_pos[0], rel_pos[1], angle])
        while len(landmark_obs) < self.num_landmarks * 3:
            landmark_obs.extend([0.0, 0.0, 0.0])
        if len(landmark_obs) > self.num_landmarks * 3:
            landmark_obs = landmark_obs[:self.num_landmarks * 3]

        obs = np.concatenate([
            obs_base,
            np.array(agent_obs, dtype=np.float32),
            np.array(landmark_obs, dtype=np.float32)
        ])
        return obs

    def global_reward(self, world):
        self.rew = 0
        landmark_observations = {l.name: set() for l in world.landmarks}

        for agent in world.agents:
            for landmark in world.landmarks:
                delta_pos = landmark.state.p_pos - agent.state.p_pos
                dist = np.linalg.norm(delta_pos)
                if dist <= agent.max_view_distance:
                    angle = np.arctan2(delta_pos[1], delta_pos[0])
                    angle_diff = (angle - agent.state.theta + np.pi) % (2 * np.pi) - np.pi
                    if abs(angle_diff) <= agent.view_angle_range:
                        landmark_observations[landmark.name].add(agent.region_id)

        for landmark in world.landmarks[:]:
            observed_regions = landmark_observations[landmark.name]
            # Count a visit only when at least two distinct agents observe it.
            if len(observed_regions) >= 2:
                new_visit = self.landmark_visits.get(landmark.name, 0) + 1
                self.landmark_visits[landmark.name] = new_visit
            else:
                self.landmark_visits[landmark.name] = 0

            if self.landmark_visits[landmark.name] >= 5:
                self.rew += 1
                world.landmarks.remove(landmark)
                del self.landmark_visits[landmark.name]

        return self.rew
