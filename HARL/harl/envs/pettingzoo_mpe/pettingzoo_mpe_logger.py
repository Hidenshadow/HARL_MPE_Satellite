from harl.common.base_logger import BaseLogger


class PettingZooMPELogger(BaseLogger):
    def get_task_name(self):
        action_type = "continuous" if self.env_args["continuous_actions"] else "discrete"
        if (
            self.env_args.get("scenario") == "simple_satellite"
            and "map_name" in self.env_args
        ):
            return f"{self.env_args['scenario']}-{self.env_args['map_name']}-{action_type}"
        return f"{self.env_args['scenario']}-{action_type}"
