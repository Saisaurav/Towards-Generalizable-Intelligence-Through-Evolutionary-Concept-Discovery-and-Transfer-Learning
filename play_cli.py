#!/usr/bin/env python3
"""Play a generated environment in the terminal - no graphics.

    python play_cli.py <env-id>                       # a saved environment from generated_envs/
    python play_cli.py generated_env_fixtures/ttt.py  # or any environment source file
    python play_cli.py <env-id> --seat 1 --opponent rule_based
    python play_cli.py <env-id> --opponent brain      # the trained brain (needs policies/brain.json)

The game is the same sandboxed Environment the web page uses; the board comes from its render_text().
At the prompt: a move number (or action name), Enter = default move for real-time games,
b = let the brain move for you, r = restart, q = quit.
"""

import argparse
import sys
from pathlib import Path
from typing import Callable

from llm_game_platform.env_store import EnvStore, EnvStoreError
from llm_game_platform.env_validator import validate_environment
from llm_game_platform.sandbox import SandboxError, SandboxProcess

_RESULT_TEXT = {"win": "You won!", "loss": "You lost.", "draw": "It's a draw."}


def resolve_source(target: str, store: EnvStore) -> Path:
    path = Path(target)
    if path.is_file():
        source = path.read_text(encoding="utf-8")
    else:
        source, _meta = store.load(target)  # raises EnvStoreError (also checks the source hash)
        path = store.source_path(target)
    report = validate_environment(source)
    if not report.ok:
        raise SystemExit("environment does not validate:\n" + "\n".join(report.errors))
    return path


def parse_move(text: str, spec: dict, legal: list[int]) -> tuple[str, int | None]:
    """Return ("move", action) / ("noop", None) / ("error", message)."""
    text = text.strip().lower()
    if not text:
        if spec.get("noop_action") is None:
            return "error", "this game needs a move; type one of: " + ", ".join(map(str, legal))
        return "noop", None
    names = [n.lower() for n in spec.get("action_names", [])]
    if text.lstrip("-").isdigit():
        action = int(text)
    elif text in names:
        action = names.index(text)
    else:
        return "error", "not a move; type one of: " + ", ".join(map(str, legal))
    if action not in legal:
        return "error", f"{action} is not allowed right now; legal moves: {', '.join(map(str, legal))}"
    return "move", action


def label(spec: dict, action: int) -> str:
    names = spec.get("action_names") or []
    return f"{action} ({names[action]})" if action < len(names) else str(action)


def play(sandbox: SandboxProcess, seat: int | None, opponent: str, policy_path: Path | None,
         input_fn: Callable[[str], str] = input, output_fn: Callable[[str], None] = print,
         max_games: int | None = None) -> list[str]:
    """Run the interactive loop; returns the human's result of each finished game."""
    spec = sandbox.spec
    if spec["action"]["type"] != "discrete":
        raise SystemExit("this environment uses continuous actions, which the text player cannot take")
    default: dict = {"kind": opponent}
    fields: dict = {}
    if policy_path is not None:
        fields["brain"] = {"policy_path": str(policy_path)}
        if opponent == "brain":
            default = {"kind": "policy", "policy_path": str(policy_path), "greedy": False}
    elif opponent == "brain":
        output_fn("(no trained brain found - the opponent will play randomly)")
        default = {"kind": "random"}
    sandbox.request("configure", default=default, **fields)
    results: list[str] = []
    output_fn(f"\n=== {spec['title']} ===\n{spec['description']}")
    seed = 1
    view = sandbox.request("reset", seed=seed, seat=seat)
    while True:
        for move in view.get("moves", []):
            for agent, action in move["actions"].items():
                output_fn(f"  player {agent} played {label(spec, action)}")
        view["moves"] = []
        output_fn("\n" + (view.get("text") or "(no text view available)"))
        if view["terminal"]:
            result = view["results"][str(view["human_seat"])]
            results.append(result)
            output_fn(_RESULT_TEXT[result])
            if max_games is not None and len(results) >= max_games:
                return results
            try:
                again = input_fn("Play again? [y/N] ").strip().lower() == "y"
            except EOFError:
                again = False
            if not again:
                return results
            seed += 1
            view = sandbox.request("reset", seed=seed, seat=view["human_seat"])
            continue
        try:
            raw = input_fn(f"Your move (player {view['human_seat']}) > ")
        except EOFError:
            return results
        command = raw.strip().lower()
        if command == "q":
            return results
        if command == "r":
            seed += 1
            view = sandbox.request("reset", seed=seed, seat=view["human_seat"])
            continue
        try:
            if command == "b":
                if policy_path is None:
                    output_fn("no trained brain available for this environment")
                    continue
                view = sandbox.request("play", action=None, brain=True)
                continue
            kind, value = parse_move(raw, spec, view["legal"])
            if kind == "error":
                output_fn(str(value))
                continue
            view = sandbox.request("play", action=value)
        except SandboxError as error:
            output_fn(f"error: {error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("environment", help="saved environment id or path to an env .py file")
    parser.add_argument("--seat", type=int, default=None, help="which player you are (default: first)")
    parser.add_argument("--opponent", choices=("random", "rule_based", "brain"), default="random")
    parser.add_argument("--policy", type=Path, default=None, help="exported brain json (default: the saved environment's)")
    args = parser.parse_args()
    store = EnvStore()
    try:
        source = resolve_source(args.environment, store)
    except EnvStoreError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    policy = args.policy
    if policy is None and not Path(args.environment).is_file():
        candidate = store.path(args.environment) / "policies" / "brain.json"
        policy = candidate if candidate.is_file() else None
    with SandboxProcess(source, call_timeout=10.0) as sandbox:
        if args.seat is not None and args.seat not in sandbox.spec["agents"]:
            print(f"error: --seat must be one of {sandbox.spec['agents']}", file=sys.stderr)
            return 1
        play(sandbox, args.seat, args.opponent, policy)
    return 0


if __name__ == "__main__":
    sys.exit(main())
