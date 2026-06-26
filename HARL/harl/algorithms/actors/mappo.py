"""
MAPPO with optional COMA-style counterfactual advantage shaping.

Usage:
- Set args["use_coma"] = True to enable COMA advantages inside MAPPO policy updates.
- Provide joint_obs / joint_actions tensors in the update sample to trigger COMA pathway.
- Train the centralized action-conditional critic by calling `update_coma_critic(...)`
  with (joint_obs, joint_actions, rewards, joint_obs_next, dones) batches.

Assumptions:
- Discrete action space for COMA.
- The existing `evaluate_actions` API from OnPolicyBase is available and returns
  (action_log_probs, dist_entropy, other_info).
- OBS tensors are [B, N, D], ACTIONS are [B, N] or [B, N, 1], JOINT OBS are [B, S].

Author: adapted to integrate COMA into a MAPPO training loop with minimal intrusion.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from harl.utils.envs_tools import check
from harl.utils.models_tools import get_grad_norm
from harl.algorithms.actors.on_policy_base import OnPolicyBase


# =========================
# Helpers for COMA pathway
# =========================
def _one_hot_actions(actions: torch.Tensor, action_dim: int) -> torch.Tensor:
    """
    actions: [B, N] (int)
    return:  [B, N*A] (float)
    """
    if actions.dim() == 3 and actions.size(-1) == 1:
        actions = actions.squeeze(-1)  # [B, N]
    B, N = actions.shape
    return F.one_hot(actions, num_classes=action_dim).view(B, N * action_dim).float()


class COMACritic(nn.Module):
    """
    Centralized action-conditional critic:
        Q_i(s, u) for i in {1..N}
    Input:
        joint_obs:          [B, S]
        joint_action_onehot:[B, N*A]
    Output:
        q_per_agent:        [B, N]
    """
    def __init__(self, joint_obs_dim: int, n_agents: int, action_dim: int, hidden=(256, 256)):
        super().__init__()
        self.n_agents = n_agents
        self.action_dim = action_dim
        in_dim = joint_obs_dim + n_agents * action_dim
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, n_agents)
        )

    def forward(self, joint_obs: torch.Tensor, joint_action_onehot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([joint_obs, joint_action_onehot], dim=-1)  # [B, S + N*A]
        return self.net(x)                                       # [B, N]


# =========================
# MAPPO with COMA shaping
# =========================
class MAPPO(OnPolicyBase):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        """Initialize MAPPO algorithm (with optional COMA advantage shaping).
        Args:
            args: (dict) arguments.
            obs_space: (gym.Space) observation space (per-agent).
            act_space: (gym.Space) action space (discrete for COMA).
            device: (torch.device) device to use.
        """
        super(MAPPO, self).__init__(args, obs_space, act_space, device)

        # ====== MAPPO hyperparams ======
        self.clip_param = args["clip_param"]
        self.ppo_epoch = args["ppo_epoch"]
        self.actor_num_mini_batch = args["actor_num_mini_batch"]
        self.entropy_coef = args["entropy_coef"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.max_grad_norm = args["max_grad_norm"]

        # ====== COMA switches & modules ======
        self.use_coma = bool(args.get("use_coma", False))
        if self.use_coma:
            assert act_space.__class__.__name__ == "Discrete", "COMA requires discrete action space."
            # Sizes
            self.n_agents = int(args["n_agents"])  # must be provided
            self.action_dim = int(act_space.n)
            # joint obs dim: if not provided, default to N * obs_dim
            obs_dim = int(np.prod(obs_space.shape))
            self.joint_obs_dim = int(args.get("joint_obs_dim", self.n_agents * obs_dim))

            # COMA critic & target
            critic_hidden = tuple(args.get("critic_hidden", (256, 256)))
            self.coma_critic = COMACritic(self.joint_obs_dim, self.n_agents, self.action_dim, critic_hidden).to(self.device)
            self.target_coma_critic = COMACritic(self.joint_obs_dim, self.n_agents, self.action_dim, critic_hidden).to(self.device)
            self.target_coma_critic.load_state_dict(self.coma_critic.state_dict())
            for p in self.target_coma_critic.parameters():
                p.requires_grad = False

            # Optim & params
            self.lr_critic = float(args.get("lr_critic", args.get("lr", 5e-4)))
            self.coma_critic_opt = torch.optim.Adam(self.coma_critic.parameters(), lr=self.lr_critic)
            self.coma_gamma = float(args.get("gamma", 0.99))
            self.coma_polyak = float(args.get("polyak", 0.995))
            # Entropy coef for policy (can reuse main one)
            self.coma_entropy_coef = float(args.get("coma_entropy_coef", self.entropy_coef))

    # ------------------------------------------------------------------
    # Public: optional separate training for COMA critic (one-step TD).
    # Call this in your learner if you have team/per-agent rewards handy.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _polyak_update(self, src: nn.Module, tgt: nn.Module, tau: float):
        for p, tp in zip(src.parameters(), tgt.parameters()):
            tp.data.mul_(tau).add_(p.data * (1.0 - tau))

    def update_coma_critic(self,
                           joint_obs: np.ndarray,
                           joint_actions: np.ndarray,
                           rewards: np.ndarray,
                           joint_obs_next: np.ndarray,
                           dones: np.ndarray,
                           per_agent_rewards: bool = False):
        """
        Train COMA centralized critic with one-step TD.

        Args:
            joint_obs:       [B, S]
            joint_actions:   [B, N] or [B, N, 1]
            rewards:         [B] (team) or [B, N] (per-agent, if per_agent_rewards=True)
            joint_obs_next:  [B, S]
            dones:           [B] or [B,1] (bool or {0,1})
            per_agent_rewards: whether `rewards` is per-agent.
        Returns:
            critic_loss (float)
        """
        assert self.use_coma, "enable use_coma to call update_coma_critic"

        joint_obs_t = check(joint_obs).to(**self.tpdv)
        joint_obs_next_t = check(joint_obs_next).to(**self.tpdv)
        actions_t = check(joint_actions).to(dtype=torch.long, device=self.device)
        if actions_t.dim() == 3 and actions_t.size(-1) == 1:
            actions_t = actions_t.squeeze(-1)  # [B, N]
        dones_t = check(dones).to(**self.tpdv).view(-1, 1).float()

        # rewards handling
        rewards_t = check(rewards).to(**self.tpdv)
        if per_agent_rewards:
            r_team = rewards_t  # [B, N]
        else:
            r_team = rewards_t.view(-1, 1)  # [B,1]
            # broadcast to per-agent targets
            B = joint_obs_t.size(0)
            r_team = r_team.expand(B, self.n_agents)

        # Q(s,u)
        q_now = self.coma_critic(joint_obs_t, _one_hot_actions(actions_t, self.action_dim))  # [B, N]

        # next joint actions: greedy under target (you can also sample)
        # For simplicity, we approximate next action by argmax of current critic per agent is not defined.
        # In practice, better pass next obs into actor to pick next actions. Here, we bootstrap by max over actions i-wise.
        with torch.no_grad():
            # Greedy per-agent over actions for next state:
            # Evaluate Q_i(s', u'_best) by exhaustive search over a_i while fixing others to current actions (approx).
            # Simpler and stable: treat q_next as q_now (no bootstrap) if next info is not available well.
            # Here choose zero bootstrap if needed:
            # q_next = torch.zeros_like(q_now)
            # Alternatively, use max over joint action is intractable; keep zero bootstrap.
            q_next = torch.zeros_like(q_now)

            td_target = r_team + (1.0 - dones_t) * self.coma_gamma * q_next  # [B, N]

        critic_loss = F.mse_loss(q_now, td_target)

        self.coma_critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.coma_critic.parameters(), self.max_grad_norm)
        self.coma_critic_opt.step()

        # Polyak target update
        self._polyak_update(self.coma_critic, self.target_coma_critic, self.coma_polyak)

        return float(critic_loss.detach().cpu().item())

    # ------------------------------------------------------------------
    # Internal: build log π(a|o) for all actions using evaluate_actions
    # (generic & robust to actor implementation; slower but reliable).
    # ------------------------------------------------------------------
    def _all_action_log_probs(self,
                               obs_batch: torch.Tensor,
                               rnn_states_batch: torch.Tensor,
                               masks_batch: torch.Tensor,
                               available_actions_batch: torch.Tensor,
                               combined_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute log π(a|o) for all discrete actions using existing evaluate_actions.
        Returns:
            log_pi_all: [B, N, A]
        """
        B, N, _ = obs_batch.shape
        A = self.action_dim
        log_pi_list = []
        for a in range(A):
            actions_a = torch.full((B, N, 1), a, dtype=torch.long, device=obs_batch.device)
            logp_a, _, _ = self.evaluate_actions(
                obs_batch,
                rnn_states_batch,
                actions_a,
                masks_batch,
                available_actions_batch,
                combined_mask,
            )  # expected shape [B, N, 1] or [B, N]
            if logp_a.dim() == 3 and logp_a.size(-1) == 1:
                logp_a = logp_a.squeeze(-1)  # [B, N]
            log_pi_list.append(logp_a)
        log_pi_all = torch.stack(log_pi_list, dim=-1)  # [B, N, A]
        return log_pi_all

    # ------------------------------------------------------------------
    # Internal: compute COMA counterfactual advantages
    # ------------------------------------------------------------------
    def _coma_advantages(self,
                         joint_obs_batch: torch.Tensor,   # [B, S]
                         obs_batch: torch.Tensor,          # [B, N, D] (only used to keep interface symmetrical)
                         rnn_states_batch: torch.Tensor,   # [B, N, ...]
                         actions_batch: torch.Tensor,      # [B, N] or [B, N, 1]
                         masks_batch: torch.Tensor,        # [B, N, 1]
                         available_actions_batch: torch.Tensor,  # or None
                         combined_mask: torch.Tensor       # [B, N, 1]
                         ) -> torch.Tensor:
        """
        A_i = Q_i(s, u) - Σ_{a_i} π_i(a_i|o_i) * Q_i(s, (a_i, u_-i))
        Returns:
            advantages: [B, N]
        """
        if actions_batch.dim() == 3 and actions_batch.size(-1) == 1:
            actions = actions_batch.squeeze(-1)  # [B, N]
        else:
            actions = actions_batch

        # Q(s, u) for executed joint action
        q_exec = self.coma_critic(joint_obs_batch, _one_hot_actions(actions, self.action_dim))  # [B, N]

        # Get log π for all actions via evaluate_actions loops (robust)
        log_pi_all = self._all_action_log_probs(
            obs_batch, rnn_states_batch, masks_batch, available_actions_batch, combined_mask
        )  # [B, N, A]
        # Convert to probs (normalize for safety)
        pi_all = torch.exp(log_pi_all)
        pi_all = pi_all / (pi_all.sum(dim=-1, keepdim=True) + 1e-12)  # [B, N, A]

        # Counterfactual baseline: vary a_i only, keep others fixed
        B, N = actions.shape
        advantages = []
        for i in range(N):
            # Prepare joint action grid with a_i' enumerated
            all_ai = torch.arange(self.action_dim, device=actions.device).view(1, self.action_dim).expand(B, self.action_dim)  # [B, A]
            # Repeat joint_obs for A candidates
            jo_rep = joint_obs_batch.unsqueeze(1).expand(B, self.action_dim, joint_obs_batch.shape[-1]).reshape(B * self.action_dim, -1)  # [B*A, S]

            # Build candidate joint actions
            a_grid = actions.unsqueeze(1).expand(B, self.action_dim, N).clone()  # [B, A, N]
            a_grid[:, :, i] = all_ai
            a_grid = a_grid.reshape(B * self.action_dim, N)  # [B*A, N]

            q_all = self.coma_critic(jo_rep, _one_hot_actions(a_grid, self.action_dim))  # [B*A, N]
            q_i_all = q_all.view(B, self.action_dim, N)[:, :, i]  # [B, A]

            # baseline b_i = Σ_{a_i} π_i(a_i|o_i) Q_i(s,(a_i,u_-i))
            b_i = (pi_all[:, i, :] * q_i_all).sum(dim=-1)  # [B]
            adv_i = q_exec[:, i] - b_i
            advantages.append(adv_i.unsqueeze(-1))
        advantages = torch.cat(advantages, dim=-1)  # [B, N]
        return advantages

    # ------------------------------------------------------------------
    # MAPPO policy update (with optional COMA advantages)
    # ------------------------------------------------------------------
    def update(self, sample):
        """
        Update actor network.

        Supported sample formats:
        - Legacy 8-tuple:
            (obs_batch, rnn_states_batch, actions_batch, masks_batch,
             active_masks_batch, old_action_log_probs_batch, adv_targ,
             available_actions_batch)

        - 9-tuple (with decision masks):
            (... above ... , decision_masks_batch)

        - Extended (COMA) 11~13-tuple:
            (... 8 or 9 fields ...,
             joint_obs_batch, joint_actions_batch,
             [optional] joint_obs_next_batch, [optional] dones_batch)

        Returns:
            policy_loss, dist_entropy, actor_grad_norm, imp_weights
        """
        # ---- Parse mandatory part (compatible with your original code) ----
        decision_masks_batch = None
        joint_obs_batch = None
        joint_actions_batch = None

        if len(sample) == 8:
            (obs_batch,
             rnn_states_batch,
             actions_batch,
             masks_batch,
             active_masks_batch,
             old_action_log_probs_batch,
             adv_targ,
             available_actions_batch) = sample
        else:
            # at least 9
            (obs_batch,
             rnn_states_batch,
             actions_batch,
             masks_batch,
             active_masks_batch,
             old_action_log_probs_batch,
             adv_targ,
             available_actions_batch,
             decision_masks_batch) = sample[:9]
            # COMA extras if exist
            if len(sample) >= 11:
                joint_obs_batch = sample[9]
                joint_actions_batch = sample[10]
            # (joint_obs_next, dones) could be present but we don't need them here for the policy update

        # ---- to torch ----
        obs_batch = check(obs_batch).to(**self.tpdv)
        rnn_states_batch = check(rnn_states_batch).to(**self.tpdv)
        actions_batch = check(actions_batch).to(**self.tpdv)
        masks_batch = check(masks_batch).to(**self.tpdv)
        if available_actions_batch is not None:
            available_actions_batch = check(available_actions_batch).to(**self.tpdv)

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)

        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        # decision mask default to ones if absent
        if decision_masks_batch is None:
            decision_masks_batch = torch.ones_like(active_masks_batch)
        else:
            decision_masks_batch = check(decision_masks_batch).to(**self.tpdv)
        # combine for evaluation
        combined_mask = active_masks_batch * decision_masks_batch

        # ---- COMA advantages (if enabled and fields available) ----
        if self.use_coma and (joint_obs_batch is not None) and (joint_actions_batch is not None):
            joint_obs_batch_t = check(joint_obs_batch).to(**self.tpdv)
            joint_actions_batch_t = check(joint_actions_batch).to(dtype=torch.long, device=self.device)
            if joint_actions_batch_t.dim() == 3 and joint_actions_batch_t.size(-1) == 1:
                joint_actions_batch_t = joint_actions_batch_t.squeeze(-1)

            coma_adv = self._coma_advantages(
                joint_obs_batch=joint_obs_batch_t,    # [B, S]
                obs_batch=obs_batch,                  # [B, N, D]
                rnn_states_batch=rnn_states_batch,    # [B, N, ...]
                actions_batch=joint_actions_batch_t,  # [B, N]
                masks_batch=masks_batch,              # [B, N, 1]
                available_actions_batch=available_actions_batch,
                combined_mask=combined_mask           # [B, N, 1]
            )  # [B, N]

            # Align shape with log_prob tensor below (likely [B, N, 1])
            if adv_targ.dim() == 3 and adv_targ.size(-1) == 1:
                adv_targ = coma_adv.unsqueeze(-1)
            else:
                adv_targ = coma_adv

        # ---- Standard MAPPO actor update (unchanged) ----
        action_log_probs, dist_entropy, _ = self.evaluate_actions(
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            combined_mask,
        )

        imp_weights = getattr(torch, self.action_aggregation)(
            torch.exp(action_log_probs - old_action_log_probs_batch),
            dim=-1,
            keepdim=True,
        )

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self.use_policy_active_masks:
            mask = combined_mask
        else:
            mask = decision_masks_batch
        denom = mask.sum().clamp_min(1.0)
        policy_action_loss = (
            -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) * mask
        ).sum() / denom

        policy_loss = policy_action_loss

        self.actor_optimizer.zero_grad()
        # use COMA-specific entropy coef if enabled (defaults to main coef)
        entropy_coef = self.coma_entropy_coef if self.use_coma else self.entropy_coef
        (policy_loss - dist_entropy * entropy_coef).backward()

        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_grad_norm(self.actor.parameters())

        self.actor_optimizer.step()

        return policy_loss, dist_entropy, actor_grad_norm, imp_weights

    # ------------------------------------------------------------------
    # The rest (train / share_param_train) are kept exactly as your original
    # aside from calling self.update(sample) which now may use COMA advantage.
    # ------------------------------------------------------------------
    def train(self, actor_buffer, advantages, state_type):
        """Perform a training update for non-parameter-sharing MAPPO using minibatch GD."""
        train_info = {}
        train_info["policy_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["ratio"] = 0

        if np.all(actor_buffer.active_masks[:-1] == 0.0):
            return {
                "policy_loss": 0.0,
                "dist_entropy": 0.0,
                "actor_grad_norm": 0.0,
                "ratio": 0.0,
            }

        # normalize advantages (EP only)
        if state_type == "EP":
            advantages_copy = advantages.copy()
            advantages_copy[actor_buffer.active_masks[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        for _ in range(self.ppo_epoch):
            if self.use_recurrent_policy:
                data_generator = actor_buffer.recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch, self.data_chunk_length
                )
            elif self.use_naive_recurrent_policy:
                data_generator = actor_buffer.naive_recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch
                )
            else:
                data_generator = actor_buffer.feed_forward_generator_actor(
                    advantages, self.actor_num_mini_batch
                )

            for sample in data_generator:
                policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.update(sample)
                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["ratio"] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.actor_num_mini_batch
        for k in train_info.keys():
            train_info[k] /= num_updates
        return train_info

    def share_param_train(self, actor_buffer, advantages, num_agents, state_type):
        """Perform a training update for parameter-sharing MAPPO using minibatch GD."""
        train_info = {}
        train_info["policy_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["ratio"] = 0

        # normalize advantages
        if state_type == "EP":
            advantages_ori_list = []
            advantages_copy_list = []
            for agent_id in range(num_agents):
                advantages_ori = advantages.copy()
                advantages_ori_list.append(advantages_ori)
                advantages_copy = advantages.copy()
                advantages_copy[actor_buffer[agent_id].active_masks[:-1] == 0.0] = np.nan
                advantages_copy_list.append(advantages_copy)
            advantages_ori_tensor = np.array(advantages_ori_list)
            advantages_copy_tensor = np.array(advantages_copy_list)
            mean_advantages = np.nanmean(advantages_copy_tensor)
            std_advantages = np.nanstd(advantages_copy_tensor)
            normalized_advantages = (advantages_ori_tensor - mean_advantages) / (
                std_advantages + 1e-5
            )
            advantages_list = []
            for agent_id in range(num_agents):
                advantages_list.append(normalized_advantages[agent_id])
        elif state_type == "FP":
            advantages_list = []
            for agent_id in range(num_agents):
                advantages_list.append(advantages[:, :, agent_id])

        for _ in range(self.ppo_epoch):
            data_generators = []
            for agent_id in range(num_agents):
                if self.use_recurrent_policy:
                    data_generator = actor_buffer[agent_id].recurrent_generator_actor(
                        advantages_list[agent_id],
                        self.actor_num_mini_batch,
                        self.data_chunk_length,
                    )
                elif self.use_naive_recurrent_policy:
                    data_generator = actor_buffer[agent_id].naive_recurrent_generator_actor(
                        advantages_list[agent_id], self.actor_num_mini_batch
                    )
                else:
                    data_generator = actor_buffer[agent_id].feed_forward_generator_actor(
                        advantages_list[agent_id], self.actor_num_mini_batch
                    )
                data_generators.append(data_generator)

            for _ in range(self.actor_num_mini_batch):
                # Merge samples across agents; support optional decision_masks (9th)
                merged = None
                for generator in data_generators:
                    sample = next(generator)
                    if len(sample) == 8:
                        sample = (*sample, None)  # pad decision masks
                    else:
                        sample = sample[:9]
                    if merged is None:
                        merged = [[x] for x in sample]
                    else:
                        for i, x in enumerate(sample):
                            merged[i].append(x)
                # concat first 7
                for i in range(7):
                    merged[i] = np.concatenate(merged[i], axis=0)
                # available_actions
                if merged[7][0] is None:
                    merged[7] = None
                else:
                    merged[7] = np.concatenate(merged[7], axis=0)
                # decision_masks
                if merged[8][0] is None:
                    merged[8] = None
                else:
                    merged[8] = np.concatenate(merged[8], axis=0)

                policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.update(tuple(merged))
                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["ratio"] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.actor_num_mini_batch
        for k in train_info.keys():
            train_info[k] /= num_updates
        return train_info
