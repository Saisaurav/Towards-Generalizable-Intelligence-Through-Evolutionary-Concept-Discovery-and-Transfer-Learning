"""Validate the hand-written environments in builtin_envs/ and install them into generated_envs/.

    python install_builtin_envs.py            # install (or refresh) all four
    python install_builtin_envs.py --list     # show the ids

They are saved in the normal EnvStore format (env.py + meta.json, no policies/), so the website's
"Load a saved environment" dropdown lists them and you can train a model for each from the UI.
Ids are fixed so the folders stay stable across re-installs. Nothing is trained here.
"""

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from llm_game_platform.env_store import EnvStore            # noqa: E402
from llm_game_platform.env_validator import validate_environment  # noqa: E402

BUILTIN = {
    "tictactoe": "Tic-Tac-Toe, two players on a 3x3 board",
    "connect4": "Connect 4, two players, 7 columns by 6 rows",
    "snake": "Snake, real-time, single player",
    "pong": "Pong, real-time, two paddles",
}


def env_id(name: str) -> str:
    return hashlib.md5(f"builtin-env:{name}".encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    store = EnvStore()
    for name, prompt in BUILTIN.items():
        if args.list:
            print(name, env_id(name))
            continue
        source = (ROOT / "builtin_envs" / f"{name}.py").read_text(encoding="utf-8")
        report = validate_environment(source)
        if not report.ok:
            print(f"{name}: validation failed: {report.errors}")
            return 1
        store.save(env_id(name), prompt=f"[built-in] {prompt}", source=source, spec=report.spec or {},
                   report=report.to_dict(), attempts=1, model="hand-written")
        print(f"{name}: installed as {env_id(name)} ({'; '.join(report.warnings) or 'no warnings'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
