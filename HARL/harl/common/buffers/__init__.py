"""Buffer registry.

EP stands for Environment Provided, as phrased by MAPPO paper.
In EP, the global states for all agents are the same.

FP stands for Feature Pruned, as phrased by MAPPO paper.
In FP, the global states for all agents are different, and thus needs the
dimension of the number of agents.
"""

from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP
from harl.common.buffers.on_policy_critic_buffer_fp import OnPolicyCriticBufferFP

CRITIC_BUFFER_REGISTRY = {
    "EP": OnPolicyCriticBufferEP,
    "FP": OnPolicyCriticBufferFP,
}

__all__ = [
    "OnPolicyActorBuffer",
    "OnPolicyCriticBufferEP",
    "OnPolicyCriticBufferFP",
    "CRITIC_BUFFER_REGISTRY",
]
