"""Tools for HARL."""
import os
import random
import numpy as np
import torch
from harl.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv


def check(value):
    """Check if value is a numpy array, if so, convert it to a torch tensor."""
    output = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
    return output


def get_shape_from_obs_space(obs_space):
    """Get shape from observation space.
    Args:
        obs_space: (gym.spaces or list) observation space
    Returns:
        obs_shape: (tuple) observation shape
    """
    if obs_space.__class__.__name__ == "Box":
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == "list":
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape


def get_shape_from_act_space(act_space):
    """Get shape from action space.
    Args:
        act_space: (gym.spaces) action space
    Returns:
        act_shape: (tuple) action shape
    """
    if act_space.__class__.__name__ == "Discrete":
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    return act_shape


def get_dim_from_act_space(act_space):
    """Get flattened action dimension from action space."""
    if act_space.__class__.__name__ == "Discrete":
        act_dim = act_space.n
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_dim = int(np.sum(act_space.nvec))
    elif act_space.__class__.__name__ == "Box":
        act_dim = int(np.prod(act_space.shape))
    elif act_space.__class__.__name__ == "MultiBinary":
        act_dim = int(np.prod(act_space.shape))
    else:
        raise NotImplementedError
    return act_dim


def make_train_env(env_name, seed, n_threads, env_args):
    """Make env for training."""
    if env_name != "pettingzoo_mpe":
        raise NotImplementedError("This release only supports pettingzoo_mpe.")

    def get_env_fn(rank):
        def init_env():
            from harl.envs.pettingzoo_mpe.pettingzoo_mpe_env import PettingZooMPEEnv

            assert env_args["scenario"] == "simple_satellite", (
                "This release only supports the simple_satellite MPE scenario."
            )
            env = PettingZooMPEEnv(env_args)
            env.seed(seed + rank * 1000)
            return env

        return init_env

    if n_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])


def make_eval_env(env_name, seed, n_threads, env_args):
    """Make env for evaluation."""
    if env_name != "pettingzoo_mpe":
        raise NotImplementedError("This release only supports pettingzoo_mpe.")

    def get_env_fn(rank):
        def init_env():
            from harl.envs.pettingzoo_mpe.pettingzoo_mpe_env import PettingZooMPEEnv

            env = PettingZooMPEEnv(env_args)
            env.seed(seed * 50000 + rank * 10000)
            return env

        return init_env

    if n_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(n_threads)])


def make_render_env(env_name, seed, env_args):
    """Make env for rendering."""
    if env_name != "pettingzoo_mpe":
        raise NotImplementedError("This release only supports pettingzoo_mpe.")

    manual_render = True  # manually call the render() function
    manual_expand_dims = True  # manually expand the num_of_parallel_envs dimension
    manual_delay = True  # manually delay the rendering by time.sleep()
    env_num = 1  # number of parallel envs
    from harl.envs.pettingzoo_mpe.pettingzoo_mpe_env import PettingZooMPEEnv

    env = PettingZooMPEEnv({**env_args, "render_mode": "human"})
    env.seed(seed * 60000)
    return env, manual_render, manual_expand_dims, manual_delay, env_num


def set_seed(args):
    """Seed the program."""
    if not args["seed_specify"]:
        args["seed"] = np.random.randint(1000, 10000)
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    os.environ["PYTHONHASHSEED"] = str(args["seed"])
    torch.manual_seed(args["seed"])
    torch.cuda.manual_seed(args["seed"])
    torch.cuda.manual_seed_all(args["seed"])


def get_num_agents(env, env_args, envs):
    """Get the number of agents in the environment."""
    if env == "pettingzoo_mpe":
        return envs.n_agents
    raise NotImplementedError("This release only supports pettingzoo_mpe.")
