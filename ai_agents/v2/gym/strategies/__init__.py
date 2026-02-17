from ai_agents.v2.gym.strategies.base_strategy import FoosballStrategy
from ai_agents.v2.gym.strategies.basic_strategy import BasicStrategy
from ai_agents.v2.gym.strategies.advanced_strategy import AdvancedStrategy

_REGISTRY = {
    "basic": BasicStrategy,
    "advanced": AdvancedStrategy,
}


def make_strategy(name: str, env, team: str) -> FoosballStrategy:
    """Instantiate a strategy by name.

    Parameters
    ----------
    name : str
        One of the registered strategy names (``"basic"``, ``"advanced"``).
    env : FoosballEnv
        The environment instance.
    team : str
        ``"y"`` or ``"b"``.
    """
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy {name!r}. Available: {list(_REGISTRY)}")
    return cls(env, team)
