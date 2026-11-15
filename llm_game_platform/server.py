"""Local HTTP API and trusted browser runtime for generated grid games."""

import json
import os
import re
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from .env_generator import EnvGenerator
from .env_service import EnvNotReady, EnvService, TRAINING_OPPONENTS
from .env_store import EnvStore, EnvStoreError
from .env_trainer import EnvTrainer
from .generator import GameGenerationError, GameGenerator
from .llm import LLMConfigurationError, LLMRequestError, OpenAICompatibleClient
from .policy import GridPolicy
from .runtime import RuntimeGenerationError, RuntimeGenerator, RuntimeSpec
from .sandbox import SandboxError
from .spec import GameEnvironment, GameSpec
from .trainer import BrainTrainer, BrainTrainingError


_WEB_ROOT = Path(__file__).with_name("web")
_GAME_ID = re.compile(r"^[0-9a-f]{32}$")
_ENV_PATH = re.compile(r"^/api/envs(?:/([0-9a-f]{32})/(status|source|reset|action|agent)|/(load))?$")


@dataclass
class _GameSession:
    spec: GameSpec | None = None
    environment: GameEnvironment | None = None
    generation_status: str = "queued"
    generation_error: str | None = None
    runtime: RuntimeSpec | None = None
    policy: GridPolicy | None = None
    runtime_status: str = "waiting"
    training_status: str = "waiting"
    runtime_error: str | None = None
    training_error: str | None = None
    runtime_repairs: list[str] | None = None


class TrainingNotReady(RuntimeError):
    pass


class GameNotReady(RuntimeError):
    pass


class GameStore:
    def __init__(self) -> None:
        self._games: dict[str, _GameSession] = {}
        self._lock = Lock()

    def create_pending(self) -> tuple[str, dict[str, Any]]:
        game_id = uuid4().hex
        game = _GameSession()
        with self._lock:
            self._games[game_id] = game
        return game_id, self._payload(game_id, game)

    def start_generation(self, game_id: str) -> None:
        with self._lock:
            self._get(game_id).generation_status = "running"

    def set_game_spec(self, game_id: str, spec: GameSpec) -> None:
        with self._lock:
            game = self._get(game_id)
            game.spec = spec
            game.environment = GameEnvironment(spec)
            game.generation_status = "ready"
            game.runtime_status = "queued"
            game.training_status = "queued"

    def fail_generation(self, game_id: str, error: str) -> None:
        with self._lock:
            game = self._get(game_id)
            game.generation_error = error
            game.generation_status = "failed"
            game.runtime_status = "skipped"
            game.training_status = "skipped"

    def reset(self, game_id: str) -> dict[str, Any]:
        with self._lock:
            game = self._get(game_id)
            self._ready_environment(game).reset()
            return self._payload(game_id, game)

    def step(self, game_id: str, action: str) -> dict[str, Any]:
        with self._lock:
            game = self._get(game_id)
            result = self._ready_environment(game).step(action)
            payload = self._payload(game_id, game)
            payload["reward"] = result["reward"]
            return payload

    def step_with_brain(self, game_id: str) -> dict[str, Any]:
        with self._lock:
            game = self._get(game_id)
            if game.policy is None:
                reason = game.training_error or f"training is {game.training_status}"
                raise TrainingNotReady(f"brain is not ready: {reason}")
            environment = self._ready_environment(game)
            action = game.policy.choose_action(environment)
            result = environment.step(action)
            payload = self._payload(game_id, game)
            payload["action"] = action
            payload["reward"] = result["reward"]
            return payload

    def set_runtime_status(self, game_id: str, status: str) -> None:
        with self._lock:
            self._get(game_id).runtime_status = status

    def set_runtime(self, game_id: str, runtime: RuntimeSpec, repairs: list[str]) -> None:
        with self._lock:
            game = self._get(game_id)
            game.runtime = runtime
            game.runtime_repairs = repairs
            game.runtime_status = "ready"

    def fail_runtime(self, game_id: str, error: str) -> None:
        with self._lock:
            game = self._get(game_id)
            game.runtime_error = error
            game.runtime_status = "failed"

    def set_training_status(self, game_id: str, status: str) -> None:
        with self._lock:
            self._get(game_id).training_status = status

    def set_policy(self, game_id: str, policy: GridPolicy) -> None:
        with self._lock:
            game = self._get(game_id)
            game.policy = policy
            game.training_status = "ready"

    def fail_training(self, game_id: str, error: str) -> None:
        with self._lock:
            game = self._get(game_id)
            game.training_error = error
            game.training_status = "failed"

    def snapshot(self, game_id: str) -> dict[str, Any]:
        with self._lock:
            return self._payload(game_id, self._get(game_id))

    @staticmethod
    def _ready_environment(game: _GameSession) -> GameEnvironment:
        if game.environment is None:
            raise GameNotReady("game rules are still being generated")
        return game.environment

    def _get(self, game_id: str) -> _GameSession:
        try:
            return self._games[game_id]
        except KeyError as error:
            raise KeyError("game not found") from error

    @staticmethod
    def _payload(game_id: str, game: _GameSession) -> dict[str, Any]:
        return {
            "id": game_id,
            "generation_status": game.generation_status,
            "generation_error": game.generation_error,
            "spec": game.spec.to_dict() if game.spec else None,
            "state": game.environment.state() if game.environment else None,
            "runtime": game.runtime.to_dict() if game.runtime else None,
            "runtime_status": game.runtime_status,
            "runtime_error": game.runtime_error,
            "runtime_repairs": game.runtime_repairs,
            "training_status": game.training_status,
            "training_error": game.training_error,
        }


class GameRequestHandler(BaseHTTPRequestHandler):
    generator = GameGenerator(OpenAICompatibleClient())
    runtime_generator = RuntimeGenerator(OpenAICompatibleClient())
    trainer = BrainTrainer()
    store = GameStore()
    executor = ThreadPoolExecutor(max_workers=4)
    # General-purpose generated-environment pipeline (LLM-written Python in a sandbox).
    env_service = EnvService(
        EnvGenerator(OpenAICompatibleClient()),
        EnvTrainer(budget=int(os.getenv("GAME_ENV_TRAINING_BUDGET", "150000"))),
        EnvStore(),
    )

    def do_GET(self) -> None:
        if self.path.startswith("/api/envs"):
            self._guarded(self._get_env)
            return
        status_match = re.fullmatch(r"/api/games/([0-9a-f]{32})/status", self.path)
        if status_match and _GAME_ID.fullmatch(status_match.group(1)):
            try:
                self._send_json(200, self.store.snapshot(status_match.group(1)))
            except KeyError as error:
                self._send_json(404, {"error": str(error).strip("'")})
            return
        # Read-only research-results viewer retained from the V8 research tree (cached data, no live LLM/training).
        if self.path in ("/research", "/research.html") or re.fullmatch(r"/data/[a-z_]+\.json", self.path):
            name = "research.html" if self.path.startswith("/research") else self.path[1:]
            try:
                body = (_WEB_ROOT / name).read_bytes()
            except OSError:
                self._send_json(404, {"error": "cached result file is unavailable; run experiments/build_frontend_data.py"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8" if name.endswith(".html") else "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/" or self.path == "/index.html":
            try:
                body = (_WEB_ROOT / "index.html").read_bytes()
            except OSError:
                self._send_json(500, {"error": "web runtime file is unavailable"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._send_json(404, {"error": "not found"})

    def _guarded(self, handler: Any) -> None:
        try:
            handler()
        except (json.JSONDecodeError, ValueError) as error:
            self._send_json(400, {"error": str(error)})
        except (KeyError, EnvStoreError) as error:
            self._send_json(404, {"error": str(error).strip("'")})
        except EnvNotReady as error:
            self._send_json(409, {"error": str(error)})
        except SandboxError as error:
            self._send_json(422, {"error": error.describe()})
        except (LLMConfigurationError, LLMRequestError) as error:
            self._send_json(503, {"error": str(error)})

    def _get_env(self) -> None:
        match = _ENV_PATH.fullmatch(self.path)
        if not match:
            self._send_json(404, {"error": "not found"})
            return
        env_id, operation, _load = match.groups()
        if env_id is None:
            self._send_json(200, {"environments": self.env_service.list_saved()})
        elif operation == "status":
            self._send_json(200, self.env_service.snapshot(env_id))
        elif operation == "source":
            body = self.env_service.source(env_id).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._send_json(404, {"error": "not found"})

    def _post_env(self) -> None:
        match = _ENV_PATH.fullmatch(self.path)
        if not match:
            self._send_json(404, {"error": "not found"})
            return
        body = self._read_json()
        env_id, operation, load = match.groups()
        service = self.env_service
        if env_id is None and load is None:
            prompt = EnvGenerator.validate_prompt(body.get("prompt"))
            session = service.create_pending(prompt, body.get("training_opponent", "random"))
            self._submit_background(self._generate_env, session.id)
            self._send_json(202, service.snapshot(session.id))
        elif load:
            loaded = service.load_saved(str(body.get("id", "")))
            self._send_json(200, service.snapshot(loaded.id))
        elif operation == "reset":
            seat = body.get("seat")
            if seat is not None and (isinstance(seat, bool) or not isinstance(seat, int)):
                raise ValueError("seat must be an integer")
            self._send_json(200, service.reset(env_id, seat, body.get("opponent")))
        elif operation == "action":
            self._send_json(200, service.play(env_id, body.get("action")))
        elif operation == "agent":
            self._send_json(200, service.play(env_id, None, use_brain=True))
        else:
            self._send_json(404, {"error": "not found"})

    def _generate_env(self, env_id: str) -> None:
        self.env_service.generate(env_id)

    def do_POST(self) -> None:
        if self.path.startswith("/api/envs"):
            self._guarded(self._post_env)
            return
        try:
            body = self._read_json()
            if self.path == "/api/games":
                prompt = self.generator.validate_prompt(body.get("prompt"))
                game_id, payload = self.store.create_pending()
                self._submit_background(self._generate_game, game_id, prompt)
                self._send_json(202, payload)
                return

            match = re.fullmatch(r"/api/games/([0-9a-f]{32})/(reset|actions|agent)", self.path)
            if not match or not _GAME_ID.fullmatch(match.group(1)):
                self._send_json(404, {"error": "not found"})
                return
            game_id, operation = match.groups()
            if operation == "reset":
                self._send_json(200, self.store.reset(game_id))
                return
            if operation == "agent":
                self._send_json(200, self.store.step_with_brain(game_id))
                return
            action = body.get("action")
            if not isinstance(action, str):
                self._send_json(400, {"error": "action must be a string"})
                return
            self._send_json(200, self.store.step(game_id, action))
        except (json.JSONDecodeError, ValueError) as error:
            self._send_json(400, {"error": str(error)})
        except KeyError as error:
            self._send_json(404, {"error": str(error).strip("'")})
        except TrainingNotReady as error:
            self._send_json(409, {"error": str(error)})
        except GameNotReady as error:
            self._send_json(409, {"error": str(error)})
        except GameGenerationError as error:
            self._send_json(
                422,
                {"error": str(error), "attempts": error.attempts, "validation_errors": error.errors},
            )
        except RuntimeGenerationError as error:
            self._send_json(
                422,
                {"error": str(error), "attempts": error.attempts, "validation_errors": error.errors},
            )
        except LLMConfigurationError as error:
            self._send_json(503, {"error": str(error)})
        except LLMRequestError as error:
            self._send_json(502, {"error": str(error)})

    def _generate_game(self, game_id: str, prompt: str) -> None:
        self.store.start_generation(game_id)
        print(f"Generating game contract for {game_id}.")
        try:
            spec, _repairs = self.generator.generate(prompt)
        except (GameGenerationError, LLMConfigurationError, LLMRequestError, ValueError) as error:
            self.store.fail_generation(game_id, str(error))
            print(f"Game contract generation failed for {game_id}: {error}")
            return
        self.store.set_game_spec(game_id, spec)
        print(f"Game contract ready for {game_id}; runtime generation and training started.")
        self._submit_background(self._generate_runtime, game_id, prompt, spec)
        self._submit_background(self._train_brain, game_id, spec)

    def _submit_background(self, worker: Any, game_id: str, *args: Any) -> None:
        future = self.executor.submit(worker, game_id, *args)
        future.add_done_callback(
            lambda completed: self._report_worker_failure(
                game_id, worker.__name__, completed
            )
        )

    def _report_worker_failure(
        self, game_id: str, worker_name: str, future: Future[Any]
    ) -> None:
        error = future.exception()
        if error is None:
            return
        message = f"{worker_name} failed unexpectedly ({type(error).__name__}): {error}"
        print(f"Background task failed for {game_id}: {message}")
        if worker_name == "_generate_game":
            self.store.fail_generation(game_id, message)
        elif worker_name == "_generate_env":
            self.env_service.fail(game_id, message)
        elif worker_name == "_generate_runtime":
            self.store.fail_runtime(game_id, message)
        elif worker_name == "_train_brain":
            self.store.fail_training(game_id, message)

    def _generate_runtime(self, game_id: str, prompt: str, spec: GameSpec) -> None:
        self.store.set_runtime_status(game_id, "running")
        try:
            runtime, repairs = self.runtime_generator.generate(prompt, spec)
        except (RuntimeGenerationError, LLMConfigurationError, LLMRequestError, ValueError) as error:
            self.store.fail_runtime(game_id, str(error))
            return
        self.store.set_runtime(game_id, runtime, repairs)

    def _train_brain(self, game_id: str, spec: GameSpec) -> None:
        self.store.set_training_status(game_id, "running")
        try:
            policy = self.trainer.train(spec)
        except (BrainTrainingError, ValueError) as error:
            self.store.fail_training(game_id, str(error))
            return
        self.store.set_policy(game_id, policy)

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("invalid Content-Length") from error
        if length < 1 or length > 16_384:
            raise ValueError("request body must be between 1 and 16384 bytes")
        body = json.loads(self.rfile.read(length))
        if not isinstance(body, dict):
            raise ValueError("request body must be a JSON object")
        return body

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")


def main() -> None:
    host = os.getenv("GAME_PLATFORM_HOST", "127.0.0.1")
    port = int(os.getenv("GAME_PLATFORM_PORT", "8000"))
    server = ThreadingHTTPServer((host, port), GameRequestHandler)
    print(f"Game platform listening at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping game platform.")
    finally:
        GameRequestHandler.env_service.close_all()
        server.server_close()


if __name__ == "__main__":
    main()
