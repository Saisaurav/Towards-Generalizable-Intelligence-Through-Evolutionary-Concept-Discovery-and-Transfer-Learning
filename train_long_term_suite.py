#!/usr/bin/env python3
"""Long-running training/evaluation orchestrator for the six built-in games.

Each trial is a fresh NEAT+PPO run using the existing EnvTrainer. Completed trials
are evaluated against random and (for multiplayer games) rule-based opponents.
The best policy per game is kept as champion.json and can be deployed to the UI.

Example: python train_long_term_suite.py --trials 3 --minutes 45
Resume completed trials: add --resume
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from llm_game_platform.env_trainer import EnvTrainer, evaluate_policy  # noqa: E402
from llm_game_platform.env_validator import validate_environment  # noqa: E402

GAMES = {
    "chess_kqk": {"population": 12, "expert_eval": True, "imitation": None},
    "chess_krk": {"population": 12, "expert_eval": True, "imitation": None},
    "pong": {"population": 8, "expert_eval": True, "imitation": 0.15},
    "tictactoe": {"population": 12, "expert_eval": True, "imitation": None},
    "snake": {"population": 8, "expert_eval": False, "imitation": 0.15},
    "connect4": {"population": 12, "expert_eval": True, "imitation": None},
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def score_metrics(metrics: dict[str, Any], single_player: bool) -> tuple[float, float]:
    if single_player:
        return float(metrics.get("mean_return", float("-inf"))), 0.0
    return (float(metrics.get("win", 0.0)) + 0.5 * float(metrics.get("draw", 0.0)),
            float(metrics.get("mean_return", 0.0)))


def run_game(name: str, args: argparse.Namespace, run_root: Path) -> None:
    source = ROOT / "builtin_envs" / f"{name}.py"
    if not source.is_file():
        raise FileNotFoundError(f"Missing environment source: {source}")

    print(f"\n{'=' * 72}\n{name.upper()} — {args.trials} trial(s), up to {args.minutes:g} min each\n{'=' * 72}", flush=True)
    report = validate_environment(source.read_text(encoding="utf-8"))
    if not report.ok or not report.spec:
        raise RuntimeError(f"Environment validation failed: {report.errors}")
    spec = report.spec
    single = len(spec["agents"]) == 1 or bool(spec.get("cooperative"))

    game_dir = run_root / name
    trials_dir = game_dir / "trials"
    policies_dir = game_dir / "policies"
    trials_dir.mkdir(parents=True, exist_ok=True)
    policies_dir.mkdir(parents=True, exist_ok=True)
    summary_path = game_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if args.resume and summary_path.exists() else {
        "game": name, "title": spec["title"], "trials": [], "champion_trial": None,
        "champion_score": None, "updated_at": None,
    }
    completed = {int(item["trial"]) for item in summary["trials"] if item.get("status") == "complete"}

    for trial in range(1, args.trials + 1):
        if args.resume and trial in completed:
            print(f"[{name}] trial {trial}: already complete; skipping", flush=True)
            continue
        seed = args.seed + (list(GAMES).index(name) * 100_000) + trial * 997
        trial_dir = trials_dir / f"trial_{trial:02d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        genome_dir = trial_dir / "genomes"
        policy_path = policies_dir / f"trial_{trial:02d}.json"
        log_path = trial_dir / "progress.jsonl"
        print(f"[{name}] trial {trial}/{args.trials} | seed={seed} | population={args.population or GAMES[name]['population']}", flush=True)

        with log_path.open("a", encoding="utf-8") as log:
            def progress(point: dict[str, float]) -> None:
                row = {"time": time.time(), **point}
                log.write(json.dumps(row, sort_keys=True) + "\n")
                log.flush()
                if "win_rate" in point:
                    print(f"  gen {int(point['gen']):>4} | steps {int(point['steps']):>8} | "
                          f"win {point['win_rate']:.1%} | W/D/L "
                          f"{int(point.get('wins', 0))}/{int(point.get('draws', 0))}/{int(point.get('losses', 0))}", flush=True)
                else:
                    print(f"  gen {int(point['gen']):>4} | steps {int(point['steps']):>8} | "
                          f"best {point['best']:.3f} | mean {point['mean']:.3f}", flush=True)

            minutes = args.minutes
            max_seconds = minutes * 60.0
            timeout = min(3600, max(600, int(max_seconds) + 600))
            os.environ["GAME_ENV_CALIBRATION_EPISODES"] = str(args.calibration_episodes)
            trainer = EnvTrainer(
                brain_binary=ROOT / "src" / ("brain.exe" if os.name == "nt" else "brain"),
                concept_bank=ROOT / "results" / "concept_bank.json",
                budget=0,
                population=args.population or GAMES[name]["population"],
                timeout=timeout,
                target_win_rate=args.target_win_rate,
                evaluation_episodes=args.eval_episodes,
                evaluation_interval_generations=args.eval_interval,
                concept_refresh_generation=0,
                max_generations=args.max_generations,
                max_seconds=max_seconds,
                quality="auto",
                head="auto",
                imitation=GAMES[name]["imitation"],
            )
            start = time.time()
            result = trainer.train_env(
                source_path=source,
                spec=spec,
                opponent="random",
                seed=seed,
                progress=progress,
                status=lambda message: print(f"  note: {message}", flush=True),
                genome_output_dir=genome_dir,
            )
            elapsed = time.time() - start
            result.save_policy(policy_path)

        eval_seed = seed + 50_000
        learner = {"kind": "policy", "policy_path": str(policy_path.resolve()), "greedy": True}
        vs_random = evaluate_policy(source, learner, "random", episodes=args.final_eval_episodes,
                                    seed=eval_seed, call_timeout=90.0)
        vs_expert = None
        if GAMES[name]["expert_eval"]:
            vs_expert = evaluate_policy(source, learner, "rule_based", episodes=args.expert_eval_episodes,
                                        seed=eval_seed + 17_000, call_timeout=90.0)
        primary, secondary = score_metrics(vs_random, single)
        trial_result = {
            "trial": trial, "seed": seed, "status": "complete", "elapsed_seconds": round(elapsed, 2),
            "training": {
                "generations": len(result.history),
                "last_generation": result.history[-1] if result.history else None,
                "target_reached": result.target_reached,
                "generation_limit_reached": result.generation_limit_reached,
                "quality_mode": result.quality_mode,
                "success_return": result.success_return,
                "calibration": result.calibration,
                "concepts_used": len(result.concepts),
                "concepts_added": result.concepts_added,
            },
            "evaluation_vs_random": vs_random,
            "evaluation_vs_rule_based": vs_expert,
            "primary_score": primary,
            "secondary_score": secondary,
            "policy_file": str(policy_path.relative_to(ROOT)).replace("\\", "/"),
        }
        write_json(trial_dir / "result.json", trial_result)
        summary["trials"] = [x for x in summary["trials"] if int(x["trial"]) != trial]
        summary["trials"].append(trial_result)
        summary["trials"].sort(key=lambda x: int(x["trial"]))

        previous = summary.get("champion_score")
        is_better = previous is None or primary > float(previous[0]) or (
            primary == float(previous[0]) and secondary > float(previous[1])
        )
        if is_better:
            champion_path = policies_dir / "champion.json"
            shutil.copy2(policy_path, champion_path)
            summary["champion_trial"] = trial
            summary["champion_score"] = [primary, secondary]
            print(f"[{name}] new champion: trial {trial} (score={primary:.4f})", flush=True)
        summary["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(summary_path, summary)

        print(f"[{name}] trial {trial} done in {elapsed / 60:.1f} min | "
              f"random W/D/L={vs_random.get('win', 0):.1%}/"
              f"{vs_random.get('draw', 0):.1%}/{vs_random.get('loss', 0):.1%} | "
              f"mean return={vs_random.get('mean_return', 0):.3f}", flush=True)

    if not (policies_dir / "champion.json").exists():
        raise RuntimeError(f"No completed policy for {name}; inspect {game_dir}")
    print(f"[{name}] champion: trial {summary.get('champion_trial')} -> {policies_dir / 'champion.json'}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", default=",".join(GAMES), help="comma-separated names or 'all'")
    parser.add_argument("--trials", type=int, default=3, help="independent runs per game (default: 3)")
    parser.add_argument("--minutes", type=float, default=45, help="maximum training minutes per trial (default: 45)")
    parser.add_argument("--population", type=int, default=0, help="override per-game population size")
    parser.add_argument("--eval-episodes", type=int, default=40, help="training checkpoint evaluation episodes")
    parser.add_argument("--eval-interval", type=int, default=5, help="evaluate every N generations")
    parser.add_argument("--final-eval-episodes", type=int, default=60)
    parser.add_argument("--expert-eval-episodes", type=int, default=30)
    parser.add_argument("--calibration-episodes", type=int, default=30)
    parser.add_argument("--target-win-rate", type=float, default=0.90)
    parser.add_argument("--max-generations", type=int, default=0, help="0 means time-limited, no generation cap")
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--output", default="training_runs/long_term_suite")
    parser.add_argument("--resume", action="store_true", help="skip trials already recorded as complete")
    args = parser.parse_args()

    if args.trials < 1 or args.minutes <= 0 or args.minutes > 55:
        parser.error("trials must be >=1 and minutes must be >0 and <=55 (brain hard timeout is 1 hour)")
    if args.population not in (0,) and not 2 <= args.population <= 128:
        parser.error("population override must be between 2 and 128")
    if not 0 < args.target_win_rate <= 1:
        parser.error("target-win-rate must be >0 and <=1")
    requested = list(GAMES) if args.games.strip().lower() == "all" else [x.strip() for x in args.games.split(",") if x.strip()]
    unknown = [name for name in requested if name not in GAMES]
    if unknown:
        parser.error(f"unknown game(s): {', '.join(unknown)}; choose from {', '.join(GAMES)}")

    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    run_root = (ROOT / args.output).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"Workspace: {ROOT}\nOutput: {run_root}\nGames: {', '.join(requested)}\n"
          f"Plan: {args.trials} trial(s) × up to {args.minutes:g} minutes per game", flush=True)
    failures = []
    for name in requested:
        try:
            run_game(name, args, run_root)
        except KeyboardInterrupt:
            print("\nInterrupted. Completed trials are saved; rerun with --resume to skip them.", flush=True)
            return 130
        except Exception as error:  # continue the suite so one bad env doesn't erase other games
            failures.append({"game": name, "error": f"{type(error).__name__}: {error}"})
            print(f"[{name}] FAILED: {failures[-1]['error']}", file=sys.stderr, flush=True)
    write_json(run_root / "suite_status.json", {
        "completed_games": [name for name in requested if (run_root / name / "policies" / "champion.json").exists()],
        "failures": failures,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    print("\nTraining suite finished." if not failures else "\nTraining suite finished with failures.", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
