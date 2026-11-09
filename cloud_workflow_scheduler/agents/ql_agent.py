"""
agents/ql_agent.py
==================
Epsilon-greedy Q-learning hyper-heuristic agent.

The agent selects a *low-level heuristic* (action) at each scheduling decision
point based on the current cloud state.  After each episode (workflow execution),
it updates the Q-table using the standard Bellman equation.

Action space
------------
0 → HEFT        (Heterogeneous Earliest Finish Time)
1 → Min-Min     (assign task to host that minimises minimum completion time)
2 → Max-Min     (assign task to host that minimises maximum completion time)
3 → RoundRobin  (cyclic host assignment)

Q-table dimensions
------------------
State : 4-tuple of discrete bins (avg_cpu, mem_pressure, queue_len, heterogeneity)
Action: one of the 4 heuristics above

Reward function
---------------
  r = -( w_e * E_norm + w_m * M_norm + w_l * LI_norm )

where
  E_norm  – normalised energy consumption
  M_norm  – normalised makespan
  LI_norm – normalised load-imbalance index
  w_e, w_m, w_l – configurable weights (default 0.4, 0.4, 0.2)
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants / type aliases
# ---------------------------------------------------------------------------

State = Tuple[int, int, int, int]
Action = int  # index into HEURISTIC_NAMES

HEURISTIC_NAMES: List[str] = ["HEFT", "Min-Min", "Max-Min", "RoundRobin"]
NUM_ACTIONS: int = len(HEURISTIC_NAMES)


# ---------------------------------------------------------------------------
# Reward calculator
# ---------------------------------------------------------------------------

class RewardCalculator:
    """
    Computes a scalar reward from raw performance metrics.

    Parameters
    ----------
    w_energy : float
        Weight for energy consumption (default 0.4).
    w_makespan : float
        Weight for makespan (default 0.4).
    w_load_imbalance : float
        Weight for load-imbalance index (default 0.2).
    baseline_energy : float
        Reference energy value for normalisation.
    baseline_makespan : float
        Reference makespan value for normalisation.
    """

    def __init__(
        self,
        w_energy: float = 0.4,
        w_makespan: float = 0.4,
        w_load_imbalance: float = 0.2,
        baseline_energy: float = 1.0,
        baseline_makespan: float = 1.0,
    ) -> None:
        assert abs(w_energy + w_makespan + w_load_imbalance - 1.0) < 1e-6, (
            "Reward weights must sum to 1.0"
        )
        self.w_e = w_energy
        self.w_m = w_makespan
        self.w_l = w_load_imbalance
        self.baseline_energy = baseline_energy
        self.baseline_makespan = baseline_makespan

    def compute(
        self,
        energy_kwh: float,
        makespan_s: float,
        load_imbalance: float,  # Already in [0, 1]
    ) -> float:
        """
        Return a scalar reward (higher is better).

        Parameters
        ----------
        energy_kwh : float
            Total energy consumed by the workflow execution.
        makespan_s : float
            Workflow makespan in seconds.
        load_imbalance : float
            Load-imbalance index in [0, 1].

        Returns
        -------
        float
            Reward value (negative cost, in range roughly (−∞, 0]).
        """
        e_norm = energy_kwh / max(self.baseline_energy, 1e-9)
        m_norm = makespan_s / max(self.baseline_makespan, 1e-9)
        li_norm = load_imbalance

        cost = self.w_e * e_norm + self.w_m * m_norm + self.w_l * li_norm
        reward = -cost
        logger.debug(
            "Reward: e_norm=%.4f m_norm=%.4f li_norm=%.4f → reward=%.4f",
            e_norm,
            m_norm,
            li_norm,
            reward,
        )
        return reward


# ---------------------------------------------------------------------------
# Q-learning agent
# ---------------------------------------------------------------------------

class QLearningAgent:
    """
    Epsilon-greedy Q-learning hyper-heuristic agent.

    Parameters
    ----------
    num_bins : int
        Number of discrete bins per state dimension (must match StateEvaluator).
    alpha : float
        Learning rate (default 0.1).
    gamma : float
        Discount factor (default 0.9).
    epsilon_start : float
        Initial exploration probability (default 1.0).
    epsilon_end : float
        Minimum exploration probability (default 0.05).
    epsilon_decay : float
        Multiplicative decay applied to epsilon after each episode (default 0.99).
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        num_bins: int = 5,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.99,
        seed: Optional[int] = None,
        reward_calculator: Optional[RewardCalculator] = None,
    ) -> None:
        self.num_bins = num_bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self._rng = random.Random(seed)
        np.random.seed(seed)

        # Q-table: DefaultDict → arrays of shape (NUM_ACTIONS,)
        self.q_table: DefaultDict[State, np.ndarray] = defaultdict(
            lambda: np.zeros(NUM_ACTIONS, dtype=float)
        )

        self.reward_calc: RewardCalculator = reward_calculator or RewardCalculator()

        # Logging / convergence tracking
        self.episode_rewards: List[float] = []
        self.action_counts: Dict[int, int] = {i: 0 for i in range(NUM_ACTIONS)}
        self._last_state: Optional[State] = None
        self._last_action: Optional[Action] = None

    # ------------------------------------------------------------------
    # Core RL API
    # ------------------------------------------------------------------

    def select_action(self, state: State) -> Action:
        """
        Choose a heuristic using epsilon-greedy policy.

        Parameters
        ----------
        state : State
            Current discretised cloud state.

        Returns
        -------
        Action
            Index into :data:`HEURISTIC_NAMES`.
        """
        if self._rng.random() < self.epsilon:
            action: Action = self._rng.randint(0, NUM_ACTIONS - 1)
            logger.debug("ε-greedy: explore → action %d (%s)", action, HEURISTIC_NAMES[action])
        else:
            action = int(np.argmax(self.q_table[state]))
            logger.debug("ε-greedy: exploit → action %d (%s)", action, HEURISTIC_NAMES[action])

        self._last_state = state
        self._last_action = action
        self.action_counts[action] += 1
        return action

    def update(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        done: bool = False,
    ) -> float:
        """
        Apply the Bellman update rule.

        Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') − Q(s,a)]

        Parameters
        ----------
        state : State
            State at which the action was taken.
        action : Action
            Action that was selected.
        reward : float
            Received reward.
        next_state : State
            State observed after action execution.
        done : bool
            Whether this is the terminal step of the episode.

        Returns
        -------
        float
            TD error (useful for monitoring convergence).
        """
        q_current = self.q_table[state][action]
        q_next = 0.0 if done else float(np.max(self.q_table[next_state]))

        td_target = reward + self.gamma * q_next
        td_error = td_target - q_current

        self.q_table[state][action] += self.alpha * td_error

        logger.debug(
            "Q-update s=%s a=%d r=%.4f → TD_err=%.4f  new_Q=%.4f",
            state, action, reward, td_error, self.q_table[state][action],
        )
        return td_error

    def end_episode(self, episode_reward: float) -> None:
        """
        Called at the end of each episode to decay epsilon and log reward.

        Parameters
        ----------
        episode_reward : float
            Total cumulative reward for the episode.
        """
        self.episode_rewards.append(episode_reward)
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        logger.info(
            "Episode %d done | reward=%.4f | ε=%.4f",
            len(self.episode_rewards),
            episode_reward,
            self.epsilon,
        )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def best_action(self, state: State) -> Tuple[Action, str]:
        """Return the greedy best action and its name for a given state."""
        action = int(np.argmax(self.q_table[state]))
        return action, HEURISTIC_NAMES[action]

    def q_values(self, state: State) -> np.ndarray:
        """Return the full Q-value vector for a given state."""
        return self.q_table[state].copy()

    def convergence_summary(self) -> Dict[str, object]:
        """
        Return a summary dict of training convergence statistics.

        Returns
        -------
        dict
            Keys: ``total_episodes``, ``final_epsilon``,
            ``mean_reward_last_10``, ``action_distribution``.
        """
        last10 = self.episode_rewards[-10:] if self.episode_rewards else [0.0]
        return {
            "total_episodes": len(self.episode_rewards),
            "final_epsilon": round(self.epsilon, 4),
            "mean_reward_last_10": round(float(np.mean(last10)), 4),
            "action_distribution": {
                HEURISTIC_NAMES[i]: self.action_counts[i]
                for i in range(NUM_ACTIONS)
            },
        }

    def save_q_table(self, filepath: str) -> None:
        """Persist the Q-table to a NumPy .npz file."""
        data = {str(k): v for k, v in self.q_table.items()}
        np.savez(filepath, **data)
        logger.info("Q-table saved to %s", filepath)

    def load_q_table(self, filepath: str) -> None:
        """Load a previously saved Q-table from a .npz file."""
        archive = np.load(filepath, allow_pickle=False)
        for key in archive.files:
            # Key format: "(b0, b1, b2, b3)" → parse back to tuple
            state: State = tuple(int(x) for x in key.strip("()").split(","))  # type: ignore[assignment]
            self.q_table[state] = archive[key]
        logger.info("Q-table loaded from %s (%d states)", filepath, len(self.q_table))

