import os
from ai_agents.common.train.impl.protagonist_antagonist_training_engine import ProtagonistAntagonistTrainingEngine
from ai_agents.common.train.impl.generic_agent_manager import GenericAgentManager
from ai_agents.common.train.impl.sac_agent import SACFoosballAgent
import sys
import argparse
from stable_baselines3.common.monitor import Monitor

from ai_agents.common.train.impl.single_player_training_engine_rule_based import SinglePlayerTrainingEngine
from ai_agents.v2.gym.full_information_protagonist_antagonist_gym_rule_based import FoosballEnv


def make_env_factory(protagonist_strategy="basic", antagonist_strategy="basic"):
    """Return a closure that creates FoosballEnv with the given strategies.

    The returned factory accepts an optional positional arg (for compatibility
    with GenericAgentManager and SinglePlayerTrainingEngine).
    """
    def factory(x=None):
        env = FoosballEnv(
            antagonist_model=None,
            protagonist_strategy_name=protagonist_strategy,
            antagonist_strategy_name=antagonist_strategy,
        )
        env = Monitor(env)
        return env
    return factory


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('-t', '--test', help='Test mode', action='store_true')
    parser.add_argument('--protagonist-strategy', default='basic',
                        choices=['basic', 'advanced', 'advanced_2'],
                        help='Strategy for protagonist (yellow)')
    parser.add_argument('--antagonist-strategy', default='basic',
                        choices=['basic', 'advanced', 'advanced_2'],
                        help='Strategy for antagonist (black)')
    args = parser.parse_args()

    env_factory = make_env_factory(args.protagonist_strategy, args.antagonist_strategy)

    model_dir = './models'
    log_dir = './logs'
    # total_epochs = 15
    total_epochs = 1
    # epoch_timesteps = int(100000)
    epoch_timesteps = int(100)

    agent_manager = GenericAgentManager(1, env_factory, SACFoosballAgent)
    agent_manager.initialize_training_agents()
    agent_manager.initialize_frozen_best_models()

    engine = SinglePlayerTrainingEngine(
        agent_manager=agent_manager,
        environment_generator=env_factory
    )

    # Start training
    if not args.test:
        engine.train(total_epochs=total_epochs, epoch_timesteps=epoch_timesteps, cycle_timesteps=10000)

    # Test the trained agent
    engine.test()
