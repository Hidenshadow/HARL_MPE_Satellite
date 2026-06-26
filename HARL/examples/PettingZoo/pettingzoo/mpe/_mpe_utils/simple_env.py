import os
import numpy as np
import pygame
import pygame.freetype
import gymnasium
from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector

region_bounds = [[-1.0, 1.0, -1.0, 1.0]]

def make_env(raw_env):
    def env(**kwargs):
        env = raw_env(**kwargs)
        if env.continuous_actions:
            env = wrappers.ClipOutOfBoundsWrapper(env)
        else:
            env = wrappers.AssertOutOfBoundsWrapper(env)
        env = wrappers.OrderEnforcingWrapper(env)
        return env
    return env

class SimpleEnv(AECEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": True,
        "render_fps": 10,
    }

    def __init__(self, scenario, world, max_cycles, render_mode=None, continuous_actions=False, local_ratio=None):
        super().__init__()
        self.render_mode = render_mode
        self.viewer = None
        self.width = 700
        self.height = 700
        self.screen = None
        self.max_size = 1
        self.game_font = None
        self.pygame_initialized = False

        self.renderOn = False
        self.seed()

        self.max_cycles = max_cycles
        self.scenario = scenario
        self.world = world
        self.continuous_actions = continuous_actions
        self.local_ratio = local_ratio

        self.scenario.reset_world(self.world, self.np_random)

        self.agents = [agent.name for agent in self.world.agents]
        self.possible_agents = self.agents[:]
        self._index_map = {agent.name: idx for idx, agent in enumerate(self.world.agents)}
        self._agent_selector = agent_selector(self.agents)

        self.action_spaces = {}
        self.observation_spaces = {}
        state_dim = 0
        for agent in self.world.agents:
            if agent.movable:
                space_dim = self.world.dim_p * 2 + 3
            elif self.continuous_actions:
                space_dim = 0
            else:
                space_dim = 1
            if not agent.silent:
                if self.continuous_actions:
                    space_dim += self.world.dim_c
                else:
                    space_dim *= self.world.dim_c

            obs_dim = len(self.scenario.observation(agent, self.world))
            state_dim += obs_dim
            if self.continuous_actions:
                self.action_spaces[agent.name] = spaces.Box(low=0, high=1, shape=(space_dim,))
            else:
                self.action_spaces[agent.name] = spaces.Discrete(space_dim)
            self.observation_spaces[agent.name] = spaces.Box(
                low=-np.float32(np.inf),
                high=+np.float32(np.inf),
                shape=(obs_dim,),
                dtype=np.float32,
            )

        self.state_space = spaces.Box(
            low=-np.float32(np.inf),
            high=+np.float32(np.inf),
            shape=(state_dim,),
            dtype=np.float32,
        )

        self.steps = 0
        self.current_actions = [None] * len(self.world.agents)

    def _init_pygame(self):
        if self.pygame_initialized:
            return
        pygame.init()
        pygame.freetype.init()
        self.screen = pygame.Surface([self.width, self.height])
        font_path = os.path.join(os.path.dirname(__file__), "secrcode.ttf")
        self.game_font = pygame.freetype.Font(font_path, 24)
        self.pygame_initialized = True

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def _region_bounds(self):
        return getattr(self.scenario, "region_bounds", region_bounds)

    def _global_bounds(self):
        global_bounds = getattr(self.scenario, "global_bounds", None)
        if global_bounds is not None:
            return global_bounds

        regions = self._region_bounds()
        return [
            min(region[0] for region in regions),
            max(region[1] for region in regions),
            min(region[2] for region in regions),
            max(region[3] for region in regions),
        ]

    def _clip_agent_to_global_bounds(self, agent):
        x_min, x_max, y_min, y_max = self._global_bounds()
        agent.state.p_pos[0] = np.clip(agent.state.p_pos[0], x_min, x_max)
        agent.state.p_pos[1] = np.clip(agent.state.p_pos[1], y_min, y_max)

    def observe(self, agent):
        return self.scenario.observation(
            self.world.agents[self._index_map[agent]], self.world
        ).astype(np.float32)

    def state(self):
        return np.concatenate([
            self.observe(agent) for agent in self.possible_agents
        ], axis=None)

    def reset(self, seed=None, return_info=False, options=None):
        if seed is not None:
            self.seed(seed=seed)
        self.scenario.reset_world(self.world, self.np_random)

        self.agents = self.possible_agents[:]
        self.rewards = {name: 0.0 for name in self.agents}
        self._cumulative_rewards = {name: 0.0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

        self.agent_selection = self._agent_selector.reset()
        self.steps = 0
        self.current_actions = [None] * len(self.world.agents)

    def _execute_world_step(self):
        for i, agent in enumerate(self.world.agents):
            action = self.current_actions[i]
            scenario_action = []
            if agent.movable:
                mdim = self.world.dim_p * 2 + 3
                if self.continuous_actions:
                    scenario_action.append(action[0:mdim])
                    action = action[mdim:]
                else:
                    scenario_action.append(action % mdim)
                    action //= mdim
            if not agent.silent:
                scenario_action.append(action)
            self._set_action(scenario_action, agent, self.action_spaces[agent.name])
            self._clip_agent_to_global_bounds(agent)

        self.world.step()
        for agent in self.world.agents:
            self._clip_agent_to_global_bounds(agent)

        global_reward = self.scenario.global_reward(self.world) if self.local_ratio is not None else 0.0
        for agent in self.world.agents:
            self.rewards[agent.name] = global_reward / len(self.world.agents)

    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        if agent.movable:
            if self.continuous_actions:
                agent.action.u[0] += action[0][1] - action[0][2]
                agent.action.u[1] += action[0][3] - action[0][4]
            else:
                if action[0] == 1:
                    agent.action.u[0] = -0.5
                elif action[0] == 2:
                    agent.action.u[0] = 0.5
                elif action[0] == 3:
                    agent.action.u[1] = -0.5
                elif action[0] == 4:
                    agent.action.u[1] = 0.5
                elif action[0] == 5:
                    agent.state.theta -= np.pi / 12
                elif action[0] == 6:
                    agent.state.theta += np.pi / 12
            sensitivity = agent.accel or 5.0
            agent.action.u *= sensitivity
            agent.state.theta = (agent.state.theta + np.pi) % (2 * np.pi) - np.pi
        if not agent.silent and not self.continuous_actions:
            agent.action.c = np.zeros(self.world.dim_c)
            agent.action.c[action[0]] = 1.0

    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return
        cur_agent = self.agent_selection
        current_idx = self._index_map[cur_agent]
        next_idx = (current_idx + 1) % len(self.world.agents)
        self.agent_selection = self._agent_selector.next()
        self.current_actions[current_idx] = action

        if next_idx == 0:
            self._execute_world_step()
            self.steps += 1
            if self.steps >= self.max_cycles:
                for a in self.agents:
                    self.truncations[a] = True
        else:
            self._clear_rewards()
        self._cumulative_rewards[cur_agent] = 0
        self._accumulate_rewards()
        if self.render_mode == "human":
            self.render()

    def enable_render(self, mode="human"):
        self._init_pygame()
        if not self.renderOn and mode == "human":
            self.screen = pygame.display.set_mode(self.screen.get_size())
            self.renderOn = True

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn("You are calling render method without specifying any render mode.")
            return
        self.enable_render(self.render_mode)
        if self.renderOn:
            pygame.event.pump()
        if self.render_mode == "human":
            self.draw()
            pygame.display.update()
        if self.render_mode == "rgb_array":
            self.draw()
            observation = np.array(pygame.surfarray.pixels3d(self.screen))
            rgb_array = np.transpose(observation, axes=(1, 0, 2))
            return rgb_array
        return None

    def draw(self):
        self.screen.fill((255, 255, 255))
        global_bounds = self._global_bounds()
        cam_range = max(2.5, max(abs(value) for value in global_bounds) * 1.25)
        scale_factor = 0.9
        half_width = int(self.width / 2 * scale_factor)
        half_height = int(self.height / 2 * scale_factor)
        for region in self._region_bounds():
            x_min, x_max, y_min, y_max = region
            screen_x_min = int((x_min / cam_range) * half_width + self.width / 2)
            screen_x_max = int((x_max / cam_range) * half_width + self.width / 2)
            screen_y_min = int((-y_max / cam_range) * half_height + self.height / 2)
            screen_y_max = int((-y_min / cam_range) * half_height + self.height / 2)
            pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(screen_x_min, screen_y_min, screen_x_max - screen_x_min, screen_y_max - screen_y_min), 1)

        for landmark in self.world.landmarks:
            lx, ly = landmark.state.p_pos
            screen_lx = int((lx / cam_range) * half_width + self.width / 2)
            screen_ly = int((-ly / cam_range) * half_height + self.height / 2)
            pygame.draw.circle(self.screen, (64, 64, 64), (screen_lx, screen_ly), 6)

        for agent in self.world.agents:
            ax, ay = agent.state.p_pos
            screen_ax = int((ax / cam_range) * half_width + self.width / 2)
            screen_ay = int((-ay / cam_range) * half_height + self.height / 2)
            radius = int(agent.size * 350)
            pygame.draw.circle(self.screen, (agent.color * 255).astype(int), (screen_ax, screen_ay), radius)
            pygame.draw.circle(self.screen, (0, 0, 0), (screen_ax, screen_ay), radius, 1)

            theta = agent.state.theta
            arrow_len = int(agent.size * 300)
            arrow_x = screen_ax + int(arrow_len * np.cos(theta))
            arrow_y = screen_ay - int(arrow_len * np.sin(theta))
            pygame.draw.line(self.screen, (50, 50, 255), (screen_ax, screen_ay), (arrow_x, arrow_y), 2)

            view_angle = agent.view_angle_range
            view_dist = agent.max_view_distance
            fan_radius = int((view_dist / cam_range) * half_width)
            fan_points = [(screen_ax, screen_ay)]
            for j in range(13):
                angle = theta - view_angle + (2 * view_angle * j / 12)
                px = screen_ax + int(fan_radius * np.cos(angle))
                py = screen_ay - int(fan_radius * np.sin(angle))
                fan_points.append((px, py))
            fan_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            pygame.draw.polygon(fan_surface, (100, 100, 255, 40), fan_points)
            self.screen.blit(fan_surface, (0, 0))
            self.game_font.render_to(self.screen, (screen_ax - 10, screen_ay - 10), agent.name, (0, 0, 0))

            for landmark in self.world.landmarks:
                rel_pos = landmark.state.p_pos - agent.state.p_pos
                dist = np.linalg.norm(rel_pos)
                if dist <= view_dist:
                    angle = np.arctan2(rel_pos[1], rel_pos[0])
                    if abs((angle - theta + np.pi) % (2 * np.pi) - np.pi) <= view_angle:
                        screen_lx = int((landmark.state.p_pos[0] / cam_range) * half_width + self.width / 2)
                        screen_ly = int((-landmark.state.p_pos[1] / cam_range) * half_height + self.height / 2)
                        pygame.draw.line(self.screen, (0, 255, 0), (screen_ax, screen_ay), (screen_lx, screen_ly), 2)
        pygame.display.flip()

    def close(self):
        if self.renderOn:
            pygame.event.pump()
            pygame.display.quit()
            self.renderOn = False
        if self.pygame_initialized:
            pygame.quit()
            self.pygame_initialized = False
