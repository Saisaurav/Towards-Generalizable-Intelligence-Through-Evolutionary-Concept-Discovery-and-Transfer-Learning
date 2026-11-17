"""Regression tests for the two reduced chess endgame environments."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_env(name):
    path = ROOT / "builtin_envs" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_endgame_contract_and_seeded_scripted_games():
    for name in ("chess_kqk", "chess_krk"):
        module = load_env(name)
        assert module.SPEC["observation_size"] == 322
        assert module.SPEC["action"]["n"] == 64
        wins = draws = 0
        for seed in range(20):
            env = module.Environment()
            env.reset(seed)
            assert len(env.observe(0)) == 322
            assert len(env.observe(1)) == 322
            while not env.is_terminal():
                agent = env.pos.turn
                legal = env.legal_actions(agent)
                assert legal
                action = env.scripted_action(agent)
                assert action in legal
                env.step({agent: action})
            result = env.result(0)
            wins += result == "win"
            draws += result == "draw"
            assert result != "loss"
        assert wins + draws == 20
        assert wins > 0
