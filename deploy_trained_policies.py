#!/usr/bin/env python3
"""Install trained champion policies into the website's saved built-in environments."""
from __future__ import annotations
import argparse
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GAMES = ("chess_kqk", "chess_krk", "pong", "tictactoe", "snake", "connect4")


def env_id(name: str) -> str:
    return hashlib.md5(f"builtin-env:{name}".encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", default=",".join(GAMES))
    parser.add_argument("--training-output", default="training_runs/long_term_suite")
    args = parser.parse_args()
    names = GAMES if args.games.strip().lower() == "all" else tuple(x.strip() for x in args.games.split(",") if x.strip())
    for name in names:
        if name not in GAMES:
            parser.error(f"unknown game: {name}")
        source = ROOT / args.training_output / name / "policies" / "champion.json"
        target_dir = ROOT / "generated_envs" / env_id(name) / "policies"
        target = target_dir / "brain.json"
        if not source.is_file():
            print(f"SKIP {name}: no champion at {source}")
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        print(f"DEPLOYED {name}: {target.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
