"""Train an algorithm."""
import argparse
import importlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PETTINGZOO = PROJECT_ROOT / "examples" / "PettingZoo"
for path in (str(PROJECT_ROOT), str(LOCAL_PETTINGZOO)):
    if path not in sys.path:
        sys.path.insert(0, path)

from harl.utils.configs_tools import get_defaults_yaml_args, update_args


def _satellite_module():
    return importlib.import_module("pettingzoo.mpe.simple_satellite")


def _print_satellite_maps():
    satellite = _satellite_module()
    for map_name in satellite.available_maps():
        config = satellite.get_satellite_config(map_name)
        rows, cols = config["grid_size"]
        print(
            f"{map_name}: size={config['size']}, agents={config['num_agents']}, "
            f"targets={config['targets']}, grid={rows}x{cols}, "
            f"episode_length={config['episode_length']}"
        )


def _sync_satellite_args(main_args, algo_args, env_args, unparsed_dict):
    if main_args["env"] != "pettingzoo_mpe":
        return
    if env_args.get("scenario") != "simple_satellite":
        return

    satellite = _satellite_module()
    config = satellite.get_satellite_config(
        env_args.get("map_name", satellite.DEFAULT_MAP_NAME)
    )
    env_args["map_name"] = config["map_name"]

    user_set_max_cycles = "max_cycles" in unparsed_dict
    user_set_episode_length = "episode_length" in unparsed_dict

    if user_set_episode_length and not user_set_max_cycles:
        env_args["max_cycles"] = int(algo_args["train"]["episode_length"])
    elif user_set_max_cycles:
        env_args["max_cycles"] = int(env_args["max_cycles"])
    else:
        env_args["max_cycles"] = config["episode_length"]

    if not user_set_episode_length:
        algo_args["train"]["episode_length"] = int(env_args["max_cycles"])


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="mappo",
        choices=[
            "happo",
            "hatrpo",
            "mappo",
            "MAE_happo",
            "MAE_hatrpo",
            "MAE_mappo",
        ],
        help="Algorithm name.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="pettingzoo_mpe",
        choices=["pettingzoo_mpe"],
        help="Environment name.",
    )
    parser.add_argument(
        "--exp_name", type=str, default="installtest", help="Experiment name."
    )
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    parser.add_argument(
        "--list_satellite_maps",
        action="store_true",
        help="List supported MPE satellite maps and exit.",
    )
    args, unparsed_args = parser.parse_known_args()
    if args.list_satellite_maps:
        _print_satellite_maps()
        return

    def process(arg):
        try:
            return eval(arg)
        except:
            return arg

    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    args = vars(args)  # convert to dict
    if args["load_config"] != "":  # load config from existing config file
        with open(args["load_config"], encoding="utf-8") as file:
            all_config = json.load(file)
        args["algo"] = all_config["main_args"]["algo"]
        args["env"] = all_config["main_args"]["env"]
        algo_args = all_config["algo_args"]
        env_args = all_config["env_args"]
    else:  # load config from corresponding yaml file
        algo_args, env_args = get_defaults_yaml_args(args["algo"], args["env"])
    update_args(unparsed_dict, algo_args, env_args)  # update args from command line
    _sync_satellite_args(args, algo_args, env_args, unparsed_dict)

    # start training
    from harl.runners import RUNNER_REGISTRY

    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
    runner.run()
    runner.close()


if __name__ == "__main__":
    main()
