"""Critic registry."""
from harl.algorithms.critics.v_critic import VCritic
from harl.algorithms.critics.v_critic_potential import VCritic_Potential

CRITIC_REGISTRY = {
    "happo": VCritic,
    "hatrpo": VCritic,
    "mappo": VCritic,
    "MAE_happo": VCritic_Potential,
    "MAE_hatrpo": VCritic_Potential,
    "MAE_mappo": VCritic_Potential,
}
