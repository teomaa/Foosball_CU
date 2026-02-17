from abc import ABC, abstractmethod
import numpy as np


class FoosballStrategy(ABC):
    """Abstract base for a foosball playing strategy.

    Parameters
    ----------
    env : FoosballEnv
        Reference to the environment (for accessing helpers and MuJoCo state).
    team : str
        ``"y"`` (yellow / protagonist, attacks +Y) or ``"b"`` (black / antagonist, attacks -Y).
    """

    def __init__(self, env, team: str):
        assert team in ("y", "b"), f"team must be 'y' or 'b', got {team!r}"
        self.env = env
        self.team = team
        # +1 for yellow (attacks +Y), -1 for black (attacks -Y)
        self.direction = 1 if team == "y" else -1

    @abstractmethod
    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        """Return an 8-d action array for this team's 4 rods."""
        ...

    @abstractmethod
    def reset(self):
        """Clear any episode-persistent state (called at env.reset())."""
        ...
