import json
import math
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request, urlopen

from llm_game_platform.generator import GameGenerationError, GameGenerator
from llm_game_platform.llm import (
    LLMConfigurationError,
    LLMRequestError,
    OpenAICompatibleClient,
)
from llm_game_platform.neat_policy import NeatPolicy
from llm_game_platform.policy import (
    GridPolicy,
    _concept_value,
    load_compatible_concepts,
    parse_expression,
)
from llm_game_platform.runtime import RuntimeGenerator, RuntimeSpec
from llm_game_platform.server import GameRequestHandler, GameStore
from llm_game_platform.spec import GameEnvironment, GameSpec


VALID_GAME = {
    "title": "Short Maze",
    "description": "Reach the star without stepping into a hazard.",
    "width": 4,
    "height": 4,
    "start": [0, 0],
    "goal": [2, 0],
    "walls": [[1, 0]],
    "hazards": [[3, 3]],
    "max_turns": 8,
}


class GameSpecTests(unittest.TestCase):
    def test_rejects_unreachable_goal_and_extra_fields(self):
        blocked = dict(VALID_GAME, walls=[[1, 0], [0, 1], [1, 1], [2, 1], [3, 1]])
        with self.assertRaisesRegex(ValueError, "reachable"):
            GameSpec.from_mapping(blocked)
        with self.assertRaisesRegex(ValueError, "unsupported fields"):
            GameSpec.from_mapping(dict(VALID_GAME, source_code="print(1)"))

    def test_environment_enforces_legal_moves_and_terminal_state(self):
        spec = GameSpec.from_mapping(VALID_GAME)
        environment = GameEnvironment(spec)
        with self.assertRaisesRegex(ValueError, "illegal action"):
            environment.step("left")
        environment.step("down")
        environment.step("right")
        environment.step("right")
        result = environment.step("up")
        self.assertEqual(result["state"]["outcome"], "won")
        self.assertEqual(result["reward"], 1.0)
        with self.assertRaisesRegex(ValueError, "already finished"):
            environment.step("up")

    def test_hazard_ends_the_episode_with_loss(self):
        spec_data = dict(VALID_GAME, walls=[], hazards=[[0, 1]], goal=[3, 3])
        environment = GameEnvironment(GameSpec.from_mapping(spec_data))
        result = environment.step("down")
        self.assertTrue(result["state"]["done"])
        self.assertEqual(result["state"]["outcome"], "lost")
        self.assertEqual(result["reward"], -1.0)


class GameGenerationTests(unittest.TestCase):
    class SequenceClient:
        def __init__(self, responses):
            self.responses = iter(responses)
            self.prompts = []

        def complete(self, system_prompt, user_prompt):
            self.prompts.append((system_prompt, user_prompt))
            return next(self.responses)

    def test_invalid_output_is_repaired_and_reported(self):
        client = self.SequenceClient(
            [
                '{"title": "bad", "width": 4}',
                json.dumps(VALID_GAME),
            ]
        )
        spec, repairs = GameGenerator(client).generate("Make a maze")
        self.assertEqual(spec.title, "Short Maze")
        self.assertEqual(len(repairs), 1)
        self.assertIn("failed validation", client.prompts[1][1])

    def test_repair_loop_is_bounded_and_exposes_validation_errors(self):
        client = self.SequenceClient(["not json", "still not json"])
        with self.assertRaises(GameGenerationError) as raised:
            GameGenerator(client, max_attempts=2).generate("Make a game")
        self.assertEqual(raised.exception.attempts, 2)
        self.assertEqual(len(raised.exception.errors), 2)
        self.assertEqual(len(client.prompts), 2)

    def test_rejects_empty_or_oversized_prompt_before_llm_call(self):
        client = self.SequenceClient([])
        generator = GameGenerator(client)
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            generator.generate(" ")
        with self.assertRaisesRegex(ValueError, "4000"):
            generator.generate("x" * 4001)
        self.assertEqual(client.prompts, [])


class LLMClientTests(unittest.TestCase):
    def test_request_timeout_is_configurable_and_validated(self):
        with patch.dict("os.environ", {"LLM_REQUEST_TIMEOUT_SECONDS": "120"}):
            self.assertEqual(OpenAICompatibleClient().timeout, 120)
        with patch.dict("os.environ", {"LLM_REQUEST_TIMEOUT_SECONDS": "invalid"}):
            with self.assertRaisesRegex(LLMConfigurationError, "positive number"):
                OpenAICompatibleClient()

    def test_http_errors_include_provider_message_without_leaking_api_key(self):
        from http.server import BaseHTTPRequestHandler

        class ProviderErrorHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {
                            "error": {
                                "code": "forbidden",
                                "details": "chat access denied; secret-key",
                            }
                        }
                    ).encode("utf-8")
                )

            def log_message(self, format, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), ProviderErrorHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = OpenAICompatibleClient(
            api_key="secret-key",
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            model="invalid-model",
        )
        try:
            with self.assertRaises(LLMRequestError) as raised:
                client.complete("system", "user")
            self.assertIn("HTTP 400", str(raised.exception))
            self.assertIn("chat access denied", str(raised.exception))
            self.assertNotIn("secret-key", str(raised.exception))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


class RuntimeGenerationTests(unittest.TestCase):
    class SequenceClient:
        def __init__(self, responses):
            self.responses = iter(responses)
            self.prompts = []

        def complete(self, system_prompt, user_prompt):
            self.prompts.append((system_prompt, user_prompt))
            return next(self.responses)

    def test_invalid_runtime_is_repaired_against_the_game_contract(self):
        runtime = {
            "theme": "forest",
            "player_symbol": "@",
            "goal_symbol": "*",
            "wall_symbol": "#",
            "hazard_symbol": "!",
            "instructions": "Reach the goal.",
        }
        client = self.SequenceClient(
            [
                json.dumps(dict(runtime, theme="javascript:alert(1)")),
                json.dumps(runtime),
            ]
        )
        spec, repairs = RuntimeGenerator(client).generate("A forest maze", GameSpec.from_mapping(VALID_GAME))
        self.assertEqual(spec.theme, "forest")
        self.assertEqual(len(repairs), 1)
        self.assertIn("Validated game contract", client.prompts[0][1])
        self.assertIn("failed validation", client.prompts[1][1])
        self.assertIn("Do not return HTML", client.prompts[0][0])


class GridPolicyTests(unittest.TestCase):
    def test_exported_policy_selects_highest_scoring_legal_action(self):
        output_base = (1 << 24) + 1
        model = {
            "n_logit": 1,
            "input_ids": list(range(14)),
            "nodes": [
                *({"id": index, "type": 0} for index in range(14)),
                {"id": 1 << 24, "type": 1},
                {"id": output_base, "type": 2},
                {"id": output_base + 1, "type": 2},
            ],
            "connections": [
                {"from": 13, "to": output_base, "weight": 1.0, "enabled": True}
            ],
            "concepts": ["action_norm"],
        }
        spec = GameSpec.from_mapping(
            dict(VALID_GAME, start=[0, 0], goal=[1, 0], walls=[], hazards=[])
        )
        environment = GameEnvironment(spec)
        self.assertEqual(GridPolicy(model).choose_action(environment), "right")

    def test_fixed_head_scores_legal_actions_without_action_dependent_concepts(self):
        output_base = (1 << 24) + 1
        model = {
            "n_logit": 3,
            "head": "fixed",
            "action_onehot": False,
            "input_ids": [0, 1],
            "nodes": [
                {"id": 0, "type": 0},
                {"id": 1, "type": 0},
                {"id": 1 << 24, "type": 1},
                *({"id": output_base + index, "type": 2} for index in range(4)),
            ],
            "connections": [
                {"from": 1 << 24, "to": output_base + 2, "weight": 2.0, "enabled": True},
                {"from": 1 << 24, "to": output_base + 3, "weight": 1.0, "enabled": True},
            ],
            "concepts": ["state_mean"],
        }
        policy = NeatPolicy(model, n_actions=3)
        self.assertEqual(policy.scores([0.25], [0, 2]), [0.0, 2.0])
        self.assertEqual(policy.choose([0.25], [0, 2]), 2)

    def test_fixed_head_rejects_mismatched_policy_output_count(self):
        with self.assertRaisesRegex(ValueError, "one logit per action"):
            NeatPolicy(
                {"n_logit": 1, "head": "fixed", "action_onehot": False,
                 "input_ids": [0], "nodes": [
                     {"id": 0, "type": 0}, {"id": 1 << 24, "type": 1},
                     {"id": (1 << 24) + 1, "type": 2}, {"id": (1 << 24) + 2, "type": 2},
                 ], "connections": [], "concepts": []},
                n_actions=2,
            )

    def test_concept_bank_filters_incompatible_primitives(self):
        bank = {
            "primitive_contract": "generic_state_action_v1",
            "concepts": [
                {"expression": "add(state_mean,action_norm)", "target_output_std": 0.25},
                {"expression": "state_boundary_margin", "target_output_std": 0.3},
                {"expression": "div(action_norm,action_norm)", "target_output_std": 2.2e-16},
                {"expression": "state_mean", "target_output_std": 0.009},
                {"expression": "state_abs_mean"},  # no stored std: now accepted because outputs are measured
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank.json"
            path.write_text(json.dumps(bank), encoding="utf-8")
            self.assertEqual(
                load_compatible_concepts(path), ["add(state_mean,action_norm)", "state_abs_mean"]
            )

    def test_v5_concepts_support_distribution_and_ordered_primitives(self):
        bank = {
            "primitive_contract": "generic_state_action_v5",
            "concepts": [
                {"expression": "state_median"},
                {"expression": "state_iqr"},
                {"expression": "state_skewness"},
                {"expression": "state_entropy"},
                {"expression": "state_position_moment"},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank.json"
            path.write_text(json.dumps(bank), encoding="utf-8")
            self.assertEqual(load_compatible_concepts(path), [
                "state_median", "state_iqr", "state_skewness", "state_entropy",
                "state_position_moment",
            ])

        self.assertAlmostEqual(
            _concept_value(parse_expression("state_entropy"), {"state_entropy": 0.8}),
            math.tanh(0.8),
        )

    def test_v6_bank_is_accepted_but_transition_features_are_not_loaded_for_grid_games(self):
        bank = {
            "primitive_contract": "generic_state_action_v6",
            "concepts": [
                {"expression": "state_mean"},
                {"expression": "mul(transition_delta_abs_mean,action_norm)"},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank.json"
            path.write_text(json.dumps(bank), encoding="utf-8")
            self.assertEqual(load_compatible_concepts(path), ["state_mean"])

    def test_v7_bank_filters_domain_specific_features_for_grid_games(self):
        bank = {
            "primitive_contract": "generic_state_action_v7",
            "concepts": [
                {"expression": "state_mean"},
                {"expression": "board_own_immediate_wins"},
                {"expression": "transition_delta_mean"},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank.json"
            path.write_text(json.dumps(bank), encoding="utf-8")
            self.assertEqual(load_compatible_concepts(path), ["state_mean"])

    def test_v4_banks_remain_loadable(self):
        bank = {
            "primitive_contract": "generic_state_action_v4",
            "concepts": [{"expression": "state_entropy"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank.json"
            path.write_text(json.dumps(bank), encoding="utf-8")
            self.assertEqual(load_compatible_concepts(path), ["state_entropy"])

    def test_v3_concepts_support_advanced_primitives_and_unary_functions(self):
        bank = {
            "primitive_contract": "generic_state_action_v3",
            "concepts": [
                {"expression": "sqrt(neg(state_min))"},
                {"expression": "log1p(state_range)"},
                {"expression": "state_boundary_margin"},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank.json"
            path.write_text(json.dumps(bank), encoding="utf-8")
            # sqrt(neg(state_min)) is dependency-compatible but exactly constant on grid states (all features >= 0, so the
            # minimum is always 0); the measured screen rejects it. Only the non-constant concept is loaded.
            self.assertEqual(load_compatible_concepts(path), ["log1p(state_range)"])
            from llm_game_platform.policy import screen_concepts
            _, rejected = screen_concepts(path)
            self.assertTrue(any(r["expression"] == "sqrt(neg(state_min))" and "standard deviation" in r["reason"] for r in rejected))

        value = _concept_value(
            parse_expression("sqrt(neg(state_min))"),
            {"state_min": -0.5},
        )
        self.assertAlmostEqual(value, math.tanh(0.5 ** 0.5))


class GameServiceTests(unittest.TestCase):
    def test_concept_api_serves_configured_bank_with_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            bank_path = Path(directory) / "bank.json"
            bank_path.write_text(json.dumps({
                "version": 7,
                "primitive_contract": "generic_state_action_v3",
                "concepts": [
                    {"id": "valid", "expression": "add(state_mean,action_norm)"},
                    {"id": "invalid", "expression": "not_a_primitive"},
                ],
            }), encoding="utf-8")
            service = type("TestEnvService", (), {
                "trainer": type("TestEnvTrainer", (), {"concept_bank": bank_path})(),
            })()
            handler = type(
                "TestConceptRequestHandler",
                (GameRequestHandler,),
                {"env_service": service},
            )
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with urlopen(f"http://127.0.0.1:{server.server_port}/api/concepts") as response:
                    payload = json.loads(response.read())
                self.assertEqual(payload["version"], 7)
                self.assertEqual(
                    [(item["id"], item["compatible"]) for item in payload["concepts"]],
                    [("valid", True), ("invalid", False)],
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_http_create_and_play_flow(self):
        class StaticClient:
            def complete(self, system_prompt, user_prompt):
                return json.dumps(VALID_GAME)

        class StaticRuntimeGenerator:
            def generate(self, prompt, spec):
                return (
                    RuntimeSpec("forest", "@", "*", "#", "!", "Reach the goal."),
                    [],
                )

        class RoutePolicy:
            def choose_action(self, environment):
                return {
                    (0, 0): "down",
                    (0, 1): "right",
                    (1, 1): "right",
                    (2, 1): "up",
                }[environment.position]

        handler = type(
            "TestGameRequestHandler",
            (GameRequestHandler,),
            {
                "generator": GameGenerator(StaticClient()),
                "runtime_generator": StaticRuntimeGenerator(),
                "trainer": type("StaticTrainer", (), {"train": lambda self, spec: RoutePolicy()})(),
                "store": GameStore(),
            },
        )
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{server.server_port}"

        def post(path, payload):
            request = Request(
                base_url + path,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request) as response:
                return json.loads(response.read())

        try:
            game = post("/api/games", {"prompt": "Make a short maze"})
            self.assertIn(game["generation_status"], {"queued", "running", "ready"})
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                status_request = Request(base_url + f"/api/games/{game['id']}/status")
                with urlopen(status_request) as response:
                    game = json.loads(response.read())
                if (
                    game["generation_status"] == "ready"
                    and game["runtime_status"] == "ready"
                    and game["training_status"] == "ready"
                ):
                    break
                time.sleep(0.01)
            self.assertEqual(game["generation_status"], "ready")
            self.assertEqual(game["runtime_status"], "ready")
            self.assertEqual(game["training_status"], "ready")
            self.assertEqual(game["state"]["outcome"], "playing")
            for _ in range(4):
                game = post(f"/api/games/{game['id']}/agent", {})
            self.assertEqual(game["state"]["outcome"], "won")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_create_returns_while_game_generation_is_still_running(self):
        started = threading.Event()
        release = threading.Event()

        class BlockingClient:
            def complete(self, system_prompt, user_prompt):
                started.set()
                if not release.wait(2):
                    raise TimeoutError("test generation was not released")
                return json.dumps(VALID_GAME)

        handler = type(
            "BlockingGameRequestHandler",
            (GameRequestHandler,),
            {
                "generator": GameGenerator(BlockingClient(), max_attempts=1),
                "store": GameStore(),
            },
        )
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{server.server_port}"
        request = Request(
            base_url + "/api/games",
            data=json.dumps({"prompt": "Make a short maze"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=1) as response:
                game = json.loads(response.read())
            self.assertEqual(response.status, 202)
            self.assertIsNone(game["spec"])
            self.assertTrue(started.wait(1))

            status_request = Request(base_url + f"/api/games/{game['id']}/status")
            with urlopen(status_request) as response:
                status = json.loads(response.read())
            self.assertIn(status["generation_status"], {"queued", "running"})
        finally:
            release.set()
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_generation_failure_is_reported_by_status(self):
        class FailingClient:
            def complete(self, system_prompt, user_prompt):
                raise RuntimeError("provider rejected the configured model")

        handler = type(
            "FailingGameRequestHandler",
            (GameRequestHandler,),
            {
                "generator": GameGenerator(FailingClient(), max_attempts=1),
                "store": GameStore(),
            },
        )
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{server.server_port}"
        create = Request(
            base_url + "/api/games",
            data=json.dumps({"prompt": "Make a short maze"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(create) as response:
                game = json.loads(response.read())
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                status_request = Request(base_url + f"/api/games/{game['id']}/status")
                with urlopen(status_request) as response:
                    game = json.loads(response.read())
                if game["generation_status"] == "failed":
                    break
                time.sleep(0.01)
            self.assertEqual(game["generation_status"], "failed")
            self.assertIn("provider rejected", game["generation_error"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
