"""Runner registry."""
from harl.runners.on_policy_ha_runner import OnPolicyHARunner
from harl.runners.on_policy_ma_runner import OnPolicyMARunner

RUNNER_REGISTRY = {
    "happo": OnPolicyHARunner,
    "hatrpo": OnPolicyHARunner,
    "mappo": OnPolicyMARunner,
    "MAE_happo": OnPolicyHARunner,
    "MAE_hatrpo": OnPolicyHARunner,
    "MAE_mappo": OnPolicyMARunner,
}
