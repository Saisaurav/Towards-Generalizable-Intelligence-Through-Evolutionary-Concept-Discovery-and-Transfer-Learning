"""Real merged server (real sandbox, real C++ `brain ext` training) with ONLY the LLM replaced by a scripted stub.

The stub returns a hand-written fixture game as the model's reply, so the generated code still goes through the
identical extract -> static check -> sandbox validation -> store -> train path as live LLM output. This is a stand-in
for the LLM only; it is NOT evidence that a live model can write valid games.
Usage: python integration_tests/e2e_stub_server.py <port> [fixture=ttt] [budget=40000] [population=16]
"""
import sys
import threading
import tempfile
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_game_platform.env_generator import EnvGenerator          # noqa: E402
from llm_game_platform.env_service import EnvService              # noqa: E402
from llm_game_platform.env_store import EnvStore                  # noqa: E402
from llm_game_platform.env_trainer import EnvTrainer              # noqa: E402
from llm_game_platform.server import GameRequestHandler           # noqa: E402
from test_generated_envs import ScriptedClient, fenced, fixture_source  # noqa: E402

port = int(sys.argv[1]); fixture = sys.argv[2] if len(sys.argv) > 2 else "ttt"
budget = int(sys.argv[3]) if len(sys.argv) > 3 else 40000; pop = int(sys.argv[4]) if len(sys.argv) > 4 else 16
root = tempfile.mkdtemp(prefix="e2e-envs-")
service = EnvService(EnvGenerator(ScriptedClient([fenced(fixture_source(fixture))]), EnvStore(root)),
                     EnvTrainer(budget=budget, population=pop, timeout=600), EnvStore(root))
handler = type("E2EHandler", (GameRequestHandler,), {"env_service": service})
server = ThreadingHTTPServer(("127.0.0.1", port), handler)
print(f"READY http://127.0.0.1:{port} fixture={fixture} budget={budget} pop={pop}", flush=True)
server.serve_forever()
