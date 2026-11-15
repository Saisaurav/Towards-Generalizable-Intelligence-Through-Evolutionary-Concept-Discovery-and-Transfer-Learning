"""Tests for the general-purpose generated-environment pipeline.

Fixtures in generated_env_fixtures/ are hand-written stand-ins for what the LLM
returns (no live LLM is available in CI). They flow through the exact same
extraction -> static check -> sandbox validation -> store -> training path.
"""

import importlib.util
import json
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from llm_game_platform import env_prompts
from llm_game_platform.env_generator import EnvGenerationError, EnvGenerator, extract_source
from llm_game_platform.env_interface import EnvSpec
from llm_game_platform.env_service import EnvService
from llm_game_platform.env_store import EnvStore, EnvStoreError
from llm_game_platform.env_trainer import EnvTrainer, evaluate_policy
from llm_game_platform.env_validator import validate_environment
from llm_game_platform.render_protocol import SceneError, validate_scene
from llm_game_platform.sandbox import SandboxError, SandboxProcess
from llm_game_platform.server import GameRequestHandler
from llm_game_platform.static_check import static_check

import itertools

import play_cli

FIXTURES = Path(__file__).parent / "generated_env_fixtures"


def fixture_source(name):
    return (FIXTURES / f"{name}.py").read_text(encoding="utf-8")


def load_fixture(name):
    """Import a fixture in-process (trusted test code) to unit-test its rules directly."""
    spec = importlib.util.spec_from_file_location(f"fx_{name}", FIXTURES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fenced(source):
    return f"Here is the game:\n```python\n{source}```\n"


class ScriptedClient:
    model = "scripted-test-model"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


MINI = '''
SPEC = {"title": "Mini", "description": "count to three", "agents": [0], "observation_size": 1,
        "action": {"type": "discrete", "n": 2}, "turn_based": True, "max_steps": 10}
class Environment:
    def reset(self, seed=None): self.n = 0
    def observe(self, agent_id): return [self.n / 3]
    def legal_actions(self, agent_id): return [] if self.is_terminal() else [0, 1]
    def active_agents(self): return [] if self.is_terminal() else [0]
    def step(self, actions):
        self.n += actions[0]
        return {0: 1.0 if self.n >= 3 else 0.0}
    def is_terminal(self): return self.n >= 3
    def result(self, agent_id): return "win"
    def render(self): return {"width": 10, "height": 10, "items": []}
'''


class InterfaceTests(unittest.TestCase):
    def test_spec_validation(self):
        spec = EnvSpec.from_mapping({
            "title": "t", "description": "d", "agents": [0, 1], "observation_size": 4,
            "action": {"type": "discrete", "n": 3}, "turn_based": True, "max_steps": 5})
        self.assertTrue(spec.trainable)
        cont = EnvSpec.from_mapping({
            "title": "t", "description": "d", "agents": [0], "observation_size": 2,
            "action": {"type": "continuous", "dim": 2, "low": -1, "high": 1}, "turn_based": False, "max_steps": 5})
        self.assertFalse(cont.trainable)
        self.assertIn("discrete", cont.untrainable_reason)
        self.assertTrue(cont.real_time)  # legacy non-turn-based specs keep their frame behavior
        realtime = EnvSpec.from_mapping({
            "title": "t", "description": "d", "agents": [0], "observation_size": 2,
            "action": {"type": "discrete", "n": 2}, "turn_based": False, "real_time": True,
            "max_steps": 0, "tick_ms": 80})
        self.assertEqual(realtime.to_dict()["tick_ms"], 80)
        self.assertTrue(realtime.to_dict()["real_time"])
        legacy_realtime = EnvSpec.from_mapping({
            "title": "t", "description": "d", "agents": [0], "observation_size": 2,
            "action": {"type": "discrete", "n": 2}, "turn_based": False, "max_steps": 1})
        self.assertTrue(legacy_realtime.real_time)
        with self.assertRaisesRegex(ValueError, "must be 0"):
            EnvSpec.from_mapping({
                "title": "t", "description": "d", "agents": [0], "observation_size": 2,
                "action": {"type": "discrete", "n": 2}, "turn_based": False, "real_time": True,
                "max_steps": 5})
        base = {"title": "t", "description": "d", "agents": [0], "observation_size": 4,
                "action": {"type": "discrete", "n": 3}, "turn_based": True, "max_steps": 5}
        for bad in (dict(base, action={"type": "discrete", "n": 500}), dict(base, extra=1),
                    dict(base, agents=[0, 0]), dict(base, max_steps=0), dict(base, learner_agent=9),
                    dict(base, real_time=True), dict(base, real_time=1)):
            with self.assertRaises(ValueError):
                EnvSpec.from_mapping(bad)

    def test_scene_validation(self):
        scene = validate_scene({"width": 10, "height": 10, "items": [
            {"type": "rect", "x": 0, "y": 0, "w": 5, "h": 5, "fill": "#fff", "action": 3, "effect": "pulse"},
            {"type": "text", "x": 1, "y": 1, "text": "hi"}]})
        self.assertEqual(scene["items"][0]["action"], 3)
        for bad in ({"width": 10}, {"width": 10, "height": 10, "items": [{"type": "blob"}]},
                    {"width": 10, "height": 10, "items": [{"type": "rect", "x": float("nan"), "y": 0, "w": 1, "h": 1}]},
                    {"width": 10, "height": 10, "items": [{"type": "rect", "x": 0, "y": 0, "w": 1, "h": 1, "fill": "url(javascript:1)"}]}):
            with self.assertRaises(SceneError):
                validate_scene(bad)

    def test_prompt_documents_interface_and_example_is_valid(self):
        self.assertIn("active_agents", env_prompts.SYSTEM_PROMPT)
        self.assertIn('SPEC["real_time"] = True', env_prompts.SYSTEM_PROMPT)
        self.assertIn('"real_time": True', env_prompts.EXAMPLE_SOURCE)
        self.assertIn("scripted_action", env_prompts.SYSTEM_PROMPT)
        self.assertNotIn("tic-tac", env_prompts.SYSTEM_PROMPT.lower())
        self.assertNotIn("tictactoe", env_prompts.SYSTEM_PROMPT.lower())  # no game template is baked in
        report = validate_environment(env_prompts.EXAMPLE_SOURCE)
        self.assertTrue(report.ok, report.errors)


class StaticCheckTests(unittest.TestCase):
    def test_rejects_dangerous_constructs(self):
        cases = {
            "import os": "import of 'os'",
            "import socket": "not allowed",
            "from subprocess import run": "not allowed",
            "x = open('/etc/passwd')": "'open'",
            "x = eval('1')": "'eval'",
            "x = ().__class__.__bases__": "dunder",
            "x = getattr(object, 'a')": "'getattr'",
        }
        for snippet, expected in cases.items():
            problems = static_check(MINI + "\n" + snippet + "\n")
            self.assertTrue(any(expected in p for p in problems), (snippet, problems))

    def test_requires_spec_class_and_methods(self):
        problems = static_check("x = 1\n")
        self.assertTrue(any("SPEC" in p for p in problems))
        self.assertTrue(any("Environment" in p for p in problems))
        problems = static_check('SPEC = {}\nclass Environment:\n    def reset(self): pass\n')
        self.assertTrue(any("observe" in p for p in problems))
        self.assertEqual(static_check(MINI), [])


class ValidatorTests(unittest.TestCase):
    def test_fixtures_pass(self):
        for name in ("ttt", "snake", "pong"):
            report = validate_environment(fixture_source(name))
            self.assertTrue(report.ok, (name, report.errors))
            self.assertGreater(report.stats["terminated"], 0)

    def failure(self, source, needle, timeout=5.0):
        report = validate_environment(source, call_timeout=timeout)
        self.assertFalse(report.ok, "expected validation failure")
        text = " ".join(report.errors)
        self.assertIn(needle, text)
        return report

    def test_wrong_observation_length(self):
        self.failure(MINI.replace("return [self.n / 3]", "return [self.n / 3, 0.0]"), "observation_size")

    def test_global_randomness_is_rejected_with_hint(self):
        src = MINI.replace("import", "import") .replace("def reset(self, seed=None): self.n = 0",
              "def reset(self, seed=None):\n        import random\n        self.n = random.randint(0, 1)")
        self.failure(src, "randint")

    def test_state_leaking_across_episodes_is_nondeterministic(self):
        src = MINI.replace("class Environment:", "COUNT = [0]\nclass Environment:").replace(
            "def observe(self, agent_id): return [self.n / 3]",
            "def observe(self, agent_id):\n        COUNT[0] += 1\n        return [min(1.0, COUNT[0] / 1e6)]")
        self.failure(src, "not deterministic")

    def test_infinite_loop_times_out(self):
        src = MINI.replace("self.n += actions[0]", "while True:\n            pass\n        self.n += actions[0]")
        self.failure(src, "exceeded", timeout=1.0)

    def test_memory_bomb_is_contained(self):
        src = MINI.replace("def reset(self, seed=None): self.n = 0", "def reset(self, seed=None):\n        self.n = 0\n        self.big = [0] * (10 ** 10)")
        report = validate_environment(src)
        self.assertFalse(report.ok)
        self.assertRegex(" ".join(report.errors).lower(), "memory|crash|exited")

    def test_never_terminating_and_zero_reward(self):
        report = self.failure(MINI.replace("return self.n >= 3", "return False"), "terminal state")
        self.assertTrue(any("max_steps" in e for e in report.errors))
        self.failure(MINI.replace("return {0: 1.0 if self.n >= 3 else 0.0}", "return {0: 0.0}"), "reward")

    def test_missing_result_and_bad_legal_actions(self):
        self.failure(MINI.replace('return "win"', 'return "victory"'), "result")
        self.failure(MINI.replace("else [0, 1]", "else [0, 5]"), "outside")

    def test_clone_must_be_independent_and_equivalent(self):
        base = MINI.replace("    def render(self)", "    def clone(self):\n        other = Environment()\n        other.n = self.n\n        return other\n    def render(self)")
        self.assertTrue(validate_environment(base).ok)
        self.failure(base.replace("other.n = self.n", "other.n = self.n + 1"), "clone()")
        shared = base.replace("def reset(self, seed=None): self.n = 0", "def reset(self, seed=None):\n        self.n = 0\n        self.hist = []").replace(
            "self.n += actions[0]", "self.hist.append(1)\n        self.n += actions[0]").replace(
            "return [self.n / 3]", "return [self.n / 3, len(self.hist) / 10]").replace(
            "other.n = self.n", "other.n = self.n\n        other.hist = self.hist").replace('"observation_size": 1', '"observation_size": 2')
        self.failure(shared, "changed the original")

    def test_runtime_errors_report_generated_code_lines(self):
        report = self.failure(MINI.replace("self.n += actions[0]", "self.n += actions[0] / 0"), "ZeroDivisionError")
        self.assertIn("line", " ".join(report.errors))


class TicTacToeRuleTests(unittest.TestCase):
    """Rules of the (LLM-style) tic-tac-toe fixture, tested directly."""

    def setUp(self):
        self.ttt = load_fixture("ttt")
        self.env = self.ttt.Environment()
        self.env.reset(1)

    def play(self, *cells):
        for cell in cells:
            self.env.step({self.env.active_agents()[0]: cell})

    def test_initial_state(self):
        self.assertEqual(self.env.legal_actions(0), list(range(9)))
        self.assertEqual(self.env.legal_actions(1), [])
        self.assertEqual(self.env.active_agents(), [0])
        self.assertFalse(self.env.is_terminal())

    def test_occupied_cells_not_selectable(self):
        self.play(4)
        self.assertNotIn(4, self.env.legal_actions(1))
        self.assertEqual(len(self.env.legal_actions(1)), 8)
        with self.assertRaises(ValueError):
            self.env.step({1: 4})

    def test_perspective_observation(self):
        self.play(0, 8)
        self.assertEqual(self.env.observe(0), [1.0] + [0.0] * 8 + [0.0] * 8 + [1.0])
        self.assertEqual(self.env.observe(1)[:9], [0.0] * 8 + [1.0])

    def test_win_detection_and_rewards(self):
        self.play(0, 3, 1, 4)
        rewards = self.env.step({0: 2})
        self.assertTrue(self.env.is_terminal())
        self.assertEqual((self.env.result(0), self.env.result(1)), ("win", "loss"))
        self.assertEqual(rewards, {0: 1.0, 1: -1.0})
        self.assertEqual(self.env.legal_actions(0), [])
        self.assertEqual(self.env.active_agents(), [])

    def test_draw_detection(self):
        self.play(0, 1, 2, 4, 3, 5, 7, 6, 8)  # X O X / O X O / O X X -> draw
        self.assertTrue(self.env.is_terminal())
        self.assertEqual((self.env.result(0), self.env.result(1)), ("draw", "draw"))

    def test_no_moves_after_termination(self):
        self.play(0, 3, 1, 4, 2)
        with self.assertRaises(ValueError):
            self.env.step({1: 5})

    def test_second_player_can_win(self):
        self.play(0, 3, 1, 4, 8)
        self.env.step({1: 5})
        self.assertEqual(self.env.result(1), "win")


class SnakeAndPongRuleTests(unittest.TestCase):
    def test_snake_movement_collision_food_growth(self):
        snake = load_fixture("snake")
        env = snake.Environment()
        env.reset(3)
        head = env.body[0]
        env.food = (head[0] + 1, head[1])             # food directly ahead (heading right)
        rewards = env.step({0: 1})
        self.assertGreater(rewards[0], 0.9)
        self.assertEqual((env.score, len(env.body)), (1, 3))
        self.assertEqual(env.body[0], (head[0] + 1, head[1]))
        self.assertNotIn(env.food, env.body)
        self.assertNotIn(3, env.legal_actions(0))      # cannot reverse into the neck
        while not env.is_terminal():                   # run into the wall
            env.step({0: 1})
        self.assertEqual(env.result(0), "loss")
        self.assertEqual(env.legal_actions(0), [])

    def test_snake_wins_only_when_board_is_full(self):
        snake = load_fixture("snake")
        env = snake.Environment()
        env.reset(3)
        self.assertEqual(snake.SPEC["max_steps"], 0)
        env.body = [(0, 0)] + [
            (x, y) for y in range(snake.H) for x in range(snake.W)
            if (x, y) not in ((0, 0), (1, 0))
        ]
        env.heading = 1
        env.food = (1, 0)
        env.step({0: 1})
        self.assertTrue(env.is_terminal())
        self.assertEqual(len(env.body), snake.W * snake.H)
        self.assertIsNone(env.food)
        self.assertEqual(env.result(0), "win")

    def test_saved_snake_has_no_frame_count_win(self):
        source = Path(__file__).parent / "generated_envs" / "bc35b744ff9244d49dad31c10fa507f1" / "env.py"
        spec = importlib.util.spec_from_file_location("saved_snake", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        env = module.Environment()
        env.reset(3)
        env.t = 500
        self.assertFalse(env.is_terminal())
        self.assertEqual(module.SPEC["max_steps"], 0)

    def test_pong_bounce_miss_and_score(self):
        pong = load_fixture("pong")
        env = pong.Environment()
        env.reset(5)
        env.paddle = [30.0, 30.0]
        env.bx, env.by, env.vx, env.vy = 8.0, 30.0, -2.0, 0.0   # heading at the left paddle
        for _ in range(3):
            env.step({0: 1, 1: 1})
        self.assertGreater(env.vx, 0)                           # bounced by the left paddle
        self.assertEqual(env.score, [0, 0])
        env.paddle = [5.0, 30.0]                                # move left paddle out of the way
        env.bx, env.by, env.vx, env.vy = 8.0, 50.0, -2.0, 0.0
        rewards = {}
        for _ in range(10):
            rewards = env.step({0: 1, 1: 1})
            if env.score[1]:
                break
        self.assertEqual(env.score, [0, 1])
        self.assertEqual(rewards[1], 1.0)
        env.bx, env.by = 30.0, 3.0                               # wall bounce
        env.vx, env.vy = 0.5, -5.0
        env.step({0: 1, 1: 1})
        self.assertGreater(env.vy, 0)
        env.score = [3, 0]
        self.assertTrue(env.is_terminal())
        self.assertEqual((env.result(0), env.result(1)), ("win", "loss"))


class SandboxRuntimeTests(unittest.TestCase):
    def test_runtime_blocks_illegal_actions_and_post_terminal_steps(self):
        with SandboxProcess(FIXTURES / "ttt.py") as box:
            self.assertEqual(box.spec["action"]["n"], 9)
            view = box.request("reset", seed=1, seat=0)
            self.assertEqual(view["legal"], list(range(9)))
            self.assertIsNotNone(view["scene"])
            view = box.request("play", action=4)
            self.assertNotIn(4, view["legal"])
            with self.assertRaises(SandboxError) as raised:
                box.request("play", action=4)
            self.assertEqual(raised.exception.kind, "contract")
            self.assertIn("illegal action", str(raised.exception))
            while not view["terminal"]:
                view = box.request("play", action=view["legal"][0])
            with self.assertRaises(SandboxError):
                box.request("play", action=0)
            self.assertIn(view["results"]["0"], ("win", "loss", "draw"))

    def test_real_time_runtime_ignores_legacy_frame_cap(self):
        source = MINI.replace('"turn_based": True, "max_steps": 10',
                              '"turn_based": False, "max_steps": 1')
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "env.py"
            path.write_text(source, encoding="utf-8")
            with SandboxProcess(path) as box:
                view = box.request("reset", seed=1, seat=0)
                view = box.request("play", action=1)
                self.assertFalse(view["terminal"])
                view = box.request("play", action=1)
                self.assertFalse(view["terminal"])
                view = box.request("play", action=1)
                self.assertTrue(view["terminal"])
                self.assertEqual(view["results"]["0"], "win")

    def test_seeded_reset_is_deterministic_in_the_sandbox(self):
        with SandboxProcess(FIXTURES / "snake.py") as box:
            a = box.request("validate_episode", seed=9, action_seed=4, check_illegal=False)
            b = box.request("validate_episode", seed=9, action_seed=4, check_illegal=False, fresh=True)
            c = box.request("validate_episode", seed=10, action_seed=4, check_illegal=False)
            self.assertEqual(a["digest"], b["digest"])
            self.assertNotEqual(a["digest"], c["digest"])

    def test_opponents_play_legally_and_rule_based_beats_random(self):
        random_vs_rule = evaluate_policy(FIXTURES / "ttt.py", "random", "rule_based", episodes=150)
        random_vs_random = evaluate_policy(FIXTURES / "ttt.py", "random", "random", episodes=150)
        self.assertLess(random_vs_rule["mean_return"], random_vs_random["mean_return"] - 0.3)
        rule_vs_random = evaluate_policy(FIXTURES / "ttt.py", "rule_based", "random", episodes=100)
        self.assertGreater(rule_vs_random["mean_return"], 0.5)  # generic look-ahead opponent (no scripted_action)
        with tempfile.TemporaryDirectory() as directory:   # generic deepcopy fallback (no clone())
            stripped = Path(directory) / "env.py"
            source = fixture_source("ttt")
            stripped.write_text(source[:source.index("    def clone")] + source[source.index("    def _full"):])
            fallback = evaluate_policy(stripped, "rule_based", "random", episodes=30)
            self.assertGreater(fallback["mean_return"], 0.4)
        pong = evaluate_policy(FIXTURES / "pong.py", "rule_based", "random", episodes=20)
        self.assertGreater(pong["win"], 0.8)                    # scripted_action hook

    def test_source_cannot_reach_secrets_or_filesystem(self):
        source = MINI.replace("def reset(self, seed=None): self.n = 0",
                              "def reset(self, seed=None):\n        import os\n        self.n = 0")
        report = validate_environment(source)
        self.assertFalse(report.ok)
        self.assertIn("os", " ".join(report.errors))


class GeneratorTests(unittest.TestCase):
    def test_extract_source(self):
        self.assertIn("SPEC", extract_source(fenced("SPEC = {}\nclass Environment: pass\n")))
        with self.assertRaises(ValueError):
            extract_source("no code here")

    def test_generation_repairs_and_persists_reproducibly(self):
        broken = fixture_source("ttt").replace('"observation_size": 18', '"observation_size": 17')
        client = ScriptedClient([fenced(broken), fenced(fixture_source("ttt"))])
        with tempfile.TemporaryDirectory() as root:
            store = EnvStore(root)
            notes = []
            generated = EnvGenerator(client, store).generate("Tic-tac-toe for two players", notes.append)
            self.assertEqual(generated.attempts, 2)
            self.assertEqual(len(generated.repairs), 1)
            self.assertIn("observation_size", generated.repairs[0])
            self.assertIn("failed validation", client.calls[1][1])      # error details reached the LLM
            self.assertIn("observation_size", client.calls[1][1])
            self.assertTrue(any("repair" in n for n in notes))
            source, meta = store.load(generated.env_id)
            self.assertEqual(source, generated.source)
            self.assertEqual(meta["model"], "scripted-test-model")
            self.assertEqual(meta["spec"]["title"], "Tic-Tac-Toe")
            self.assertEqual([e["id"] for e in store.list()], [generated.env_id])
            (store.path(generated.env_id) / "env.py").write_text(source + "# tampered\n")
            with self.assertRaises(EnvStoreError):
                store.load(generated.env_id)

    def test_repair_loop_is_bounded(self):
        client = ScriptedClient([fenced(MINI.replace("is_terminal(self): return self.n >= 3", "is_terminal(self): return False")), "plain text"])
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(EnvGenerationError) as raised:
                EnvGenerator(client, EnvStore(root), max_attempts=2).generate("anything")
        self.assertEqual(raised.exception.attempts, 2)
        self.assertEqual(len(client.calls), 2)

    def test_prompt_limits(self):
        generator = EnvGenerator(ScriptedClient(["x"]), EnvStore(tempfile.gettempdir()))
        with self.assertRaises(ValueError):
            generator.generate(" ")
        with self.assertRaises(ValueError):
            generator.generate("x" * 4001)


class TextModeTests(unittest.TestCase):
    """The no-graphics path: render_text() views and the terminal player."""

    def test_fixtures_provide_text_views(self):
        for name, needle in (("ttt", "type a cell number"), ("snake", "score"), ("pong", "Moves:")):
            with SandboxProcess(FIXTURES / f"{name}.py") as box:
                self.assertTrue(box.spec["has_render_text"], name)
                text = box.request("reset", seed=1, seat=0)["text"]
                self.assertIn(needle, text)

    def test_missing_render_text_falls_back_and_warns(self):
        report = validate_environment(MINI)
        self.assertTrue(report.ok)
        self.assertTrue(any("render_text" in w for w in report.warnings))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "env.py"
            path.write_text(MINI.replace('"items": []}', '"items": [{"type": "text", "x": 1, "y": 1, "text": "hello"}], "hud": "status"}'))
            with SandboxProcess(path) as box:
                self.assertEqual(box.request("reset", seed=1)["text"], "hello\nstatus")

    def test_bad_render_text_is_a_validation_error(self):
        source = MINI.replace("    def render(self)", "    def render_text(self):\n        return 5\n    def render(self)")
        report = validate_environment(source)
        self.assertFalse(report.ok)
        self.assertIn("render_text", " ".join(report.errors))

    def test_cli_plays_a_full_game_and_rejects_bad_input(self):
        outputs, cells = [], itertools.cycle(range(9))
        script = itertools.chain(["", "99", "zzz"], (str(c) for c in cells))

        def fake_input(prompt):
            if prompt.startswith("Play again"):
                return "n"
            return next(script)

        with SandboxProcess(FIXTURES / "ttt.py") as box:
            results = play_cli.play(box, 0, "random", None, fake_input, outputs.append, max_games=1)
        text = "\n".join(outputs)
        self.assertEqual(len(results), 1)
        self.assertIn(results[0], ("win", "loss", "draw"))
        self.assertIn("this game needs a move", text)       # empty input on a turn-based game
        self.assertIn("is not allowed right now", text)      # 99
        self.assertIn("not a move", text)                     # zzz
        self.assertRegex(text, "You won!|You lost.|It's a draw.")
        self.assertIn("---+---+---", text)                    # the board was printed

    def test_cli_names_quit_restart_and_brain_without_policy(self):
        inputs = iter(["center", "b", "r", "q"])
        outputs = []
        with SandboxProcess(FIXTURES / "ttt.py") as box:
            play_cli.play(box, 0, "rule_based", None, lambda _p: next(inputs), outputs.append)
        text = "\n".join(outputs)
        self.assertIn("player 1 played", text)                # the opponent answered "center"
        self.assertIn("no trained brain available", text)
        self.assertIn("X", text)

    def test_cli_default_move_for_real_time_games_and_resolving_sources(self):
        outputs = []
        inputs = itertools.chain(["", "", "2"], itertools.repeat("q"))
        with SandboxProcess(FIXTURES / "snake.py") as box:
            play_cli.play(box, None, "random", None, lambda _p: next(inputs), outputs.append)
        self.assertIn("Moves: 0=up", "\n".join(outputs))
        store = EnvStore(tempfile.mkdtemp())
        self.assertTrue(play_cli.resolve_source(str(FIXTURES / "ttt.py"), store).is_file())
        with self.assertRaises(EnvStoreError):
            play_cli.resolve_source("0" * 32, store)


def small_trainer(budget=30000, population=16):
    return EnvTrainer(budget=budget, population=population, timeout=300, min_generations=1)


class LearnerIntegrationTests(unittest.TestCase):
    """C++ NEAT+PPO learner driving sandboxed generated environments via the relay."""

    def train(self, name, opponent="random", budget=30000, seed=3):
        path = FIXTURES / f"{name}.py"
        with SandboxProcess(path) as box:
            spec = box.spec
        return path, spec, small_trainer(budget).train_env(path, spec, opponent, seed=seed)

    def test_training_stops_when_configured_win_rate_is_met(self):
        source = MINI.replace("else [0, 1]", "else [1]")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "always-win.py"
            path.write_text(source, encoding="utf-8")
            with SandboxProcess(path) as sandbox:
                spec = sandbox.spec
            trainer = EnvTrainer(
                budget=0,
                population=2,
                timeout=60,
                target_win_rate=0.9,
                evaluation_episodes=5,
                evaluation_interval_generations=1,
            )
            result = trainer.train_env(path, spec, seed=17, concepts=[])
        self.assertTrue(result.target_reached)
        self.assertEqual(result.history[-1]["win_rate"], 1.0)

    def test_ttt_trains_and_python_policy_matches_cpp_heldout(self):
        path, spec, result = self.train("ttt", budget=60000)
        self.assertGreater(len(result.history), 1)
        self.assertEqual(result.opponent, "random")
        self.assertTrue(result.concepts)                       # transferred concepts are the only carry-over
        policy_file = result.save_policy(Path(tempfile.mkdtemp()) / "p.json")
        mine = evaluate_policy(path, {"kind": "policy", "policy_path": str(policy_file), "greedy": False},
                               "random", episodes=3000, seed=99)
        self.assertLess(abs(mine["mean_return"] - result.heldout_return), 0.15)
        baseline = evaluate_policy(path, "random", "random", episodes=3000, seed=99)
        self.assertAlmostEqual(baseline["mean_return"], 0.0, delta=0.1)

    def test_every_training_run_starts_from_scratch(self):
        _, _, first = self.train("ttt", budget=8000, seed=5)
        _, _, second = self.train("ttt", budget=8000, seed=5)
        self.assertEqual(first.history[0], second.history[0])  # same seed -> same fresh start
        _, _, other = self.train("ttt", budget=8000, seed=6)
        self.assertNotEqual(first.history[0]["best"], other.history[0]["best"])

    def test_snake_and_pong_execute_and_train_without_errors(self):
        # Environment execution + learner plumbing only; no claim about final skill.
        for name in ("snake", "pong"):
            _, spec, result = self.train(name, budget=15000)
            self.assertTrue(result.history, name)
            self.assertIsNotNone(result.heldout_return, name)

    def test_training_against_rule_based_and_frozen_policy(self):
        path, spec, result = self.train("ttt", opponent="rule_based", budget=6000)
        self.assertEqual(result.opponent, "rule_based")
        with tempfile.TemporaryDirectory() as directory:
            rounds = small_trainer(6000, 12).train_self_play(path, spec, Path(directory), rounds=2)
            self.assertEqual([r.opponent for r in rounds], ["random", "policy"])
            self.assertTrue((Path(directory) / "selfplay_round_1.json").is_file())

    def test_broken_environment_surfaces_a_clear_training_error(self):
        broken = MINI.replace("self.n += actions[0]", "self.n += actions[0]\n        if self.n == 2: raise RuntimeError('boom')")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "env.py"
            path.write_text(broken)
            with SandboxProcess(path) as box:
                spec = box.spec
            from llm_game_platform.trainer import BrainTrainingError
            with self.assertRaisesRegex(BrainTrainingError, "boom"):
                small_trainer(5000).train_env(path, spec, "random")

    def test_continuous_actions_run_but_are_not_trainable(self):
        source = MINI.replace('"action": {"type": "discrete", "n": 2}', '"action": {"type": "continuous", "dim": 1, "low": 0, "high": 1}') \
            .replace("self.n += actions[0]", "self.n += int(actions[0][0] * 2 + 1)").replace("else [0, 1]", "else []")
        report = validate_environment(source)
        self.assertTrue(report.ok, report.errors)
        self.assertTrue(any("discrete" in w for w in report.warnings))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "env.py"
            path.write_text(source)
            with SandboxProcess(path) as box:
                spec = box.spec
            from llm_game_platform.trainer import BrainTrainingError
            with self.assertRaisesRegex(BrainTrainingError, "discrete"):
                small_trainer().train_env(path, spec, "random")


class EnvApiTests(unittest.TestCase):
    def start(self, replies, root):
        client = ScriptedClient(replies)
        service = EnvService(EnvGenerator(client, EnvStore(root)), small_trainer(6000, 8), EnvStore(root))
        handler = type("EnvTestHandler", (GameRequestHandler,), {"env_service": service})
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close(), service.close_all()))
        return f"http://127.0.0.1:{server.server_port}", service

    @staticmethod
    def call(url, data=None):
        request = Request(url, data=None if data is None else json.dumps(data).encode(), method="GET" if data is None else "POST",
                          headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=20) as response:
                body = response.read()
                return response.status, (json.loads(body) if response.headers["Content-Type"].startswith("application/json") else body.decode())
        except HTTPError as error:
            return error.code, json.loads(error.read())

    def wait(self, base, env_id, done):
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            _, snap = self.call(f"{base}/api/envs/{env_id}/status")
            if done(snap):
                return snap
            time.sleep(0.2)
        self.fail("timed out waiting for environment")

    def test_full_flow_generate_train_play_reload(self):
        with tempfile.TemporaryDirectory() as root:
            base, service = self.start([fenced(fixture_source("ttt"))], root)
            status, snap = self.call(f"{base}/api/envs", {"prompt": "tic tac toe", "training_opponent": "random"})
            self.assertEqual(status, 202)
            env_id = snap["id"]
            snap = self.wait(base, env_id, lambda s: s["training_status"] in ("ready", "failed"))
            self.assertEqual(snap["generation_status"], "ready")
            self.assertEqual(snap["training_status"], "ready", snap["training_error"])
            self.assertEqual(snap["spec"]["title"], "Tic-Tac-Toe")
            self.assertTrue(snap["training_history"] and snap["concepts"])
            self.assertTrue(snap["brain_ready"])

            status, snap = self.call(f"{base}/api/envs/{env_id}/reset", {"seat": 0, "opponent": "rule_based"})
            self.assertEqual(status, 200)
            view = snap["view"]
            self.assertEqual(view["legal"], list(range(9)))
            self.assertTrue(any(i.get("action") == 4 for i in view["scene"]["items"]))   # clickable board
            status, snap = self.call(f"{base}/api/envs/{env_id}/action", {"action": 4})
            self.assertEqual(status, 200)
            self.assertNotIn(4, snap["view"]["legal"])
            status, err = self.call(f"{base}/api/envs/{env_id}/action", {"action": 4})
            self.assertEqual(status, 400)
            self.assertIn("illegal", err["error"])
            status, snap = self.call(f"{base}/api/envs/{env_id}/agent", {})        # brain plays the human seat
            self.assertEqual(status, 200)
            status, snap = self.call(f"{base}/api/envs/{env_id}/reset", {"seat": 1, "opponent": "brain"})
            self.assertEqual(status, 200)
            self.assertEqual(snap["human_seat"], 1)
            status, text = self.call(f"{base}/api/envs/{env_id}/source")
            self.assertEqual((status, text), (200, fixture_source("ttt")))
            status, listing = self.call(f"{base}/api/envs")
            self.assertEqual([e["id"] for e in listing["environments"]], [env_id])

            # a fresh service can reload the saved environment and its trained policy
            service2 = EnvService(EnvGenerator(ScriptedClient(["x"]), EnvStore(root)), small_trainer(), EnvStore(root))
            reloaded = service2.load_saved(env_id)
            self.assertEqual(reloaded.training_status, "ready")
            self.assertEqual(service2.reset(env_id, 0)["view"]["legal"], list(range(9)))
            service2.close_all()

    def test_generation_failure_and_validation(self):
        with tempfile.TemporaryDirectory() as root:
            base, _ = self.start(["I cannot do that"], root)
            status, snap = self.call(f"{base}/api/envs", {"prompt": "something"})
            snap = self.wait(base, snap["id"], lambda s: s["generation_status"] in ("ready", "failed"))
            self.assertEqual(snap["generation_status"], "failed")
            self.assertIn("python code block", snap["generation_error"])
            self.assertEqual(self.call(f"{base}/api/envs", {"prompt": ""})[0], 400)
            self.assertEqual(self.call(f"{base}/api/envs/{'0' * 32}/status")[0], 404)

    def test_real_time_game_steps_with_noop(self):
        with tempfile.TemporaryDirectory() as root:
            base, _ = self.start([fenced(fixture_source("snake"))], root)
            status, snap = self.call(f"{base}/api/envs", {"prompt": "snake"})
            env_id = snap["id"]
            self.wait(base, env_id, lambda s: s["generation_status"] == "ready" and s["training_status"] in ("ready", "failed"))
            status, snap = self.call(f"{base}/api/envs/{env_id}/reset", {})
            before = snap["view"]["steps"]
            status, snap = self.call(f"{base}/api/envs/{env_id}/action", {"action": None})
            self.assertEqual(snap["view"]["steps"], before + 1)
            self.assertTrue(snap["spec"]["hold_last_action"])

    def test_real_time_learner_advances_one_frame_per_action(self):
        source = MINI.replace(
            '"turn_based": True, "max_steps": 10',
            '"turn_based": False, "real_time": True, "max_steps": 0, "tick_ms": 80',
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "realtime.py"
            path.write_text(source, encoding="utf-8")
            with SandboxProcess(path) as sandbox:
                sandbox.raw_line("R 7")
                fields = sandbox.raw_line("A 1").split()
        legal_count = int(fields[3])
        state_size_index = 4 + legal_count
        self.assertEqual(int(fields[state_size_index]), 1)
        self.assertAlmostEqual(float(fields[state_size_index + 1]), 1.0 / 3.0)


SNAKE_SOURCE = (FIXTURES / "snake.py").read_text(encoding="utf-8")


def _run_learner_episode(source: str, limit: int = 400):
    """Drive the brain-mode text protocol with a constant action; return (frames, final outcome field)."""
    with tempfile.TemporaryDirectory() as root:
        path = Path(root) / "env.py"
        path.write_text(source, encoding="utf-8")
        with SandboxProcess(path) as box:
            line = box.raw_line("R 4")
            for frames in range(1, limit + 1):
                fields = line.split()
                line = box.raw_line(f"A {fields[4]}")
                fields = line.split()
                if int(fields[2]):
                    return frames, int(fields[-1])
    return None, None


class RealTimeSafetyAndLearnabilityTests(unittest.TestCase):
    IMMORTAL = SNAKE_SOURCE.replace(
        "return not (0 <= x < W and 0 <= y < H) or cell in self.body[:-1]", "return False")

    def test_spec_accepts_frame_limits_only_for_real_time(self):
        base = dict(title="t", description="d", agents=[0], observation_size=1,
                    action={"type": "discrete", "n": 2}, turn_based=False, max_steps=0)
        spec = EnvSpec.from_mapping({**base, "max_frames": 50, "stall_frames": 10})
        self.assertEqual((spec.frame_cap, spec.stall_frames), (50, 10))
        with self.assertRaises(ValueError):
            EnvSpec.from_mapping({**base, "turn_based": True, "max_steps": 5, "max_frames": 50})

    def test_frame_cap_truncates_a_game_that_never_ends(self):
        source = self.IMMORTAL.replace('"stall_frames": 150,', '"stall_frames": 0,').replace('"max_frames": 1500', '"max_frames": 40')
        frames, outcome = _run_learner_episode(source)
        self.assertEqual(frames, 40)
        self.assertEqual(outcome, 0)  # truncation is a draw, never a win

    def test_stall_limit_truncates_when_return_stops_improving(self):
        source = (self.IMMORTAL.replace('"stall_frames": 150', '"stall_frames": 10')
                  .replace("reward += 0.1 if new_dist < old_dist else -0.1", "reward += 0.0"))
        frames, outcome = _run_learner_episode(source)
        self.assertEqual(frames, 10)
        self.assertEqual(outcome, 0)

    def test_gate_rejects_reward_that_punishes_competent_play(self):
        # Real LLM output: -0.01 per frame makes the scripted policy (which survives) score far below random.
        source = (Path(__file__).parent / "generated_envs" / "d5acd95bfdaa4722aa8bbc0f462ee6b8" / "env.py").read_text(encoding="utf-8")
        report = validate_environment(source)
        self.assertFalse(report.ok)
        self.assertTrue(any("learnability" in e and "worse than random" in e for e in report.errors), report.errors)

    def test_updated_snake_fixture_passes_the_gate_without_warnings(self):
        report = validate_environment(SNAKE_SOURCE)
        self.assertTrue(report.ok, report.errors)
        self.assertEqual(report.warnings, [])
        info = report.stats["learnability"]
        self.assertGreater(info["scripted_mean_return"], info["random_mean_return"] + 5)
        self.assertEqual(info["quality_mode"], "return")

    def test_quality_mode_is_return_for_snake_and_result_for_pong(self):
        trainer = small_trainer()
        for name, mode in (("snake", "return"), ("pong", "result")):
            path = FIXTURES / name
            path = path.with_suffix(".py")
            with SandboxProcess(path) as box:
                spec = box.spec
            got, bar, _ = trainer._quality_plan(path, spec, {"kind": "random"})
            self.assertEqual(got, mode, name)
            self.assertEqual(bar is not None, mode == "return", name)

    def test_generation_cap_stops_gracefully_and_still_returns_a_policy(self):
        trainer = EnvTrainer(budget=0, population=12, timeout=300, max_generations=2,
                             evaluation_episodes=10, evaluation_interval_generations=1)
        path = FIXTURES / "snake.py"
        with SandboxProcess(path) as box:
            spec = box.spec
        result = trainer.train_env(path, spec, "random", seed=2)
        self.assertEqual(len(result.history), 2)
        self.assertEqual(result.quality_mode, "return")
        self.assertIsNotNone(result.heldout_return)
        self.assertIsNotNone(result.policy)

    def test_fixed_head_filters_action_and_transition_concepts(self):
        from llm_game_platform.neat_policy import fixed_head_concepts
        kept = fixed_head_concepts(["state_mean", "add(action_norm, state_var)", "transition_delta_mean",
                                    "mul(state_max, state_min)", "action_wins_now"])
        self.assertEqual(kept, ["state_mean", "mul(state_max, state_min)"])

    def test_head_selection_and_python_policy_parity_inputs(self):
        # auto => fixed head for real-time games, per-action head for turn-based ones.
        snake = FIXTURES / "snake.py"
        with SandboxProcess(snake) as box:
            spec = box.spec
        trainer = EnvTrainer(budget=0, population=12, timeout=300, max_generations=2,
                             evaluation_episodes=10, evaluation_interval_generations=1)
        result = trainer.train_env(snake, spec, "random", seed=4)       # default concept bank
        self.assertEqual(result.model["head"], "fixed")
        self.assertEqual(result.model["n_logit"], spec["action"]["n"])
        legal = [0, 1, 3]
        obs = [0.0] * spec["observation_size"]
        scores = result.policy.scores(obs, legal)
        self.assertEqual(len(scores), len(legal))
        self.assertIn(result.policy.choose(obs, legal), legal)
        ttt = FIXTURES / "ttt.py"
        with SandboxProcess(ttt) as box:
            ttt_spec = box.spec
        other = EnvTrainer(budget=3000, population=8, timeout=300).train_env(ttt, ttt_spec, "random", seed=4)
        self.assertEqual(other.model["head"], "per_action")


if __name__ == "__main__":
    unittest.main()
