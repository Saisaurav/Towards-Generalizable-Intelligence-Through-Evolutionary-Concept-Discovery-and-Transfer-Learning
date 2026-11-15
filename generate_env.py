#!/usr/bin/env python3
"""Generate, validate and (optionally) train an environment from a natural-language prompt.

    export LLM_API_KEY=...            # same settings as the web app (see llm_game_platform/llm.py)
    python generate_env.py "tic-tac-toe for two players" --train --opponent rule_based
    python generate_env.py --acceptance ttt        # use a prompt from generated_env_fixtures/acceptance_prompts.json

The generated source and metadata are saved under generated_envs/<id>/ (or $GENERATED_ENVS_DIR).
"""

import argparse
import json
import os
import sys
from pathlib import Path

from llm_game_platform.env_generator import EnvGenerationError, EnvGenerator
from llm_game_platform.env_store import EnvStore
from llm_game_platform.env_trainer import EnvTrainer, evaluate_policy
from llm_game_platform.llm import OpenAICompatibleClient


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompt", nargs="?")
    parser.add_argument("--acceptance", choices=("ttt", "snake", "pong"))
    parser.add_argument("--train", action="store_true", help="train a fresh learner after validation")
    parser.add_argument("--opponent", default="random", choices=("random", "rule_based"))
    parser.add_argument(
        "--budget",
        type=int,
        default=int(os.getenv("GAME_ENV_MAX_STEPS", os.getenv("GAME_ENV_TRAINING_BUDGET", "0"))),
        help="maximum environment steps (0 runs until the win-rate target or time limit)",
    )
    parser.add_argument("--attempts", type=int, default=4)
    args = parser.parse_args()
    prompt = args.prompt
    if args.acceptance:
        prompt = json.loads((Path(__file__).parent / "generated_env_fixtures" / "acceptance_prompts.json").read_text())[args.acceptance]
    if not prompt:
        parser.error("give a prompt or --acceptance")
    store = EnvStore()
    try:
        generated = EnvGenerator(OpenAICompatibleClient(), store, max_attempts=args.attempts).generate(
            prompt, progress=lambda message: print(f"  {message}", flush=True))
    except EnvGenerationError as error:
        print(f"FAILED after {error.attempts} attempts:\n" + "\n".join(error.errors), file=sys.stderr)
        return 1
    print(f"OK: {generated.spec['title']} -> {store.path(generated.env_id)} ({generated.attempts} attempt(s))")
    print(json.dumps(generated.report.stats, indent=2))
    for warning in generated.report.warnings:
        print("warning:", warning)
    if args.train:
        trainer = EnvTrainer(budget=args.budget)
        result = trainer.train_env(store.source_path(generated.env_id), generated.spec, args.opponent,
                                   progress=lambda p: print(f"  gen {p['gen']:3d} steps {p['steps']:7d} best {p['best']:.2f} mean {p['mean']:.2f}", flush=True))
        policy = result.save_policy(store.path(generated.env_id) / "policies" / "brain.json")
        print("held-out return (sampled):", result.heldout_return)
        for opponent in ("random", "rule_based"):
            score = evaluate_policy(store.source_path(generated.env_id), {"kind": "policy", "policy_path": str(policy), "greedy": True}, opponent, 300)
            print(f"vs {opponent}:", json.dumps({k: round(v, 3) for k, v in score.items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
