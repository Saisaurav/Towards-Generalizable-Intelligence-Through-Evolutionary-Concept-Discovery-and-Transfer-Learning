"""Small, additive training service for the two built-in robotics surrogate environments.

Training is performed by the same C++ NEAT+PPO brain used by the project. The browser
visualizer is a deterministic 2-D surrogate; this service does not launch Gazebo.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
ROBOTICS = ROOT / "robotics_sim"
POLICIES = ROBOTICS / "policies"
RUNS = ROBOTICS / "runs"

_PROGRESS = re.compile(r"gen\s+(\d+)\s+steps\s+(\d+)\s+best\s+(-?[\d.]+)\s+mean\s+(-?[\d.]+)")
_HELDOUT = re.compile(r"HELDOUT success\s+([\d.]+)\s+mean_return\s+(-?[\d.]+)\s+episodes\s+(\d+)")


class RoboticsTrainingManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="robot-training")
        self._jobs: dict[str, dict[str, Any]] = {}
        self._active: dict[str, str] = {}
        ROBOTICS.mkdir(parents=True, exist_ok=True)
        POLICIES.mkdir(parents=True, exist_ok=True)
        RUNS.mkdir(parents=True, exist_ok=True)
        (ROBOTICS / "none.txt").touch(exist_ok=True)

    @staticmethod
    def _settings(kind: str) -> tuple[int, str]:
        if kind == "turtlebot":
            return 25_000_000, "rnoise=0.05,rdelay=2,rlag=1.5"
        if kind == "arm2link":
            return 8_000_000, "rnoise=0.05,rdelay=2,rlag=1.5,feat=3,fk=0.25"
        raise ValueError("kind must be 'turtlebot' or 'arm2link'")

    def _binary(self) -> Path:
        exe = "brain_robotics.exe" if os.name == "nt" else "brain_robotics"
        binary = SRC / exe
        sources = [SRC / "main.cpp", SRC / "robot_envs.h", SRC / "robotics_env.h", SRC / "neat.h", SRC / "genome.h"]
        newest = max((p.stat().st_mtime for p in sources if p.exists()), default=0)
        if binary.exists() and binary.stat().st_mtime >= newest:
            return binary
        compiler = shutil.which("g++") or shutil.which("clang++")
        if not compiler:
            raise RuntimeError("Robotics trainer is not built. Install a C++17 compiler, then run: cd src && make brain")
        cmd = [compiler, "-std=c++17", "-O2", "-pthread", str(SRC / "main.cpp"), "-o", str(binary)]
        result = subprocess.run(cmd, cwd=SRC, capture_output=True, text=True, timeout=180)
        if result.returncode:
            raise RuntimeError("Could not build robotics brain: " + (result.stderr[-2500:] or result.stdout[-2500:]))
        return binary

    def status(self, kind: str) -> dict[str, Any]:
        self._settings(kind)
        with self._lock:
            job_id = self._active.get(kind)
            job = dict(self._jobs[job_id]) if job_id else None
        policy_path = POLICIES / f"{kind}.json"
        policy_meta: dict[str, Any] | None = None
        if policy_path.exists():
            try:
                p = json.loads(policy_path.read_text(encoding="utf-8"))
                policy_meta = {"env": p.get("env"), "act_dim": p.get("act_dim"),
                               "inputs": len(p.get("input_ids", [])), "nodes": len(p.get("nodes", [])),
                               "connections": len(p.get("connections", [])),
                               "obs_names": p.get("obs_names", []), "env_args": p.get("env_args", "")}
            except (OSError, json.JSONDecodeError):
                policy_meta = None
        return {"kind": kind, "job": job, "policy_available": policy_meta is not None,
                "policy": policy_meta, "default_budget": self._settings(kind)[0]}

    def start(self, kind: str, budget: int | None = None, seed: int = 3) -> dict[str, Any]:
        default_budget, env_args = self._settings(kind)
        if budget is None:
            budget = default_budget
        if isinstance(budget, bool) or not isinstance(budget, int) or not 100_000 <= budget <= 50_000_000:
            raise ValueError("budget must be an integer between 100,000 and 50,000,000 environment steps")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**31 - 1:
            raise ValueError("seed must be an integer between 0 and 2,147,483,647")
        with self._lock:
            active_id = self._active.get(kind)
            if active_id and self._jobs[active_id]["status"] == "running":
                raise RuntimeError(f"{kind} training is already running")
            job_id = f"{kind}-{uuid.uuid4().hex[:10]}-{seed}"
            job = {"id": job_id, "kind": kind, "status": "queued", "budget": budget,
                   "seed": seed, "steps": 0, "generation": 0, "best": None, "mean": None,
                   "success": None, "mean_return": None, "log": [], "error": None,
                   "started_at": None, "finished_at": None}
            self._jobs[job_id] = job
            self._active[kind] = job_id
        self._executor.submit(self._run, job_id, env_args)
        return dict(job)

    def _append_log(self, job_id: str, line: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job["log"].append(line.rstrip())
            job["log"] = job["log"][-120:]
            m = _PROGRESS.search(line)
            if m:
                job["generation"], job["steps"] = int(m.group(1)), int(m.group(2))
                job["best"], job["mean"] = float(m.group(3)), float(m.group(4))
            m = _HELDOUT.search(line)
            if m:
                job["success"], job["mean_return"] = float(m.group(1)), float(m.group(2))

    def _run(self, job_id: str, env_args: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job["status"] = "building"
            job["started_at"] = time.time()
        try:
            binary = self._binary()
            with self._lock:
                job["status"] = "running"
            kind = job["kind"]
            run_dir = RUNS / job_id
            genome_dir = run_dir / "genomes"
            run_dir.mkdir(parents=True, exist_ok=True)
            policy_path = POLICIES / f"{kind}.json"
            cmd = [str(binary), kind, str(ROBOTICS / "none.txt"), str(job["budget"]),
                   str(job["seed"]), "48", str(policy_path), "--preset", "lam",
                   "--env-args", env_args, "--genome-dir", str(genome_dir)]
            with (run_dir / "training.log").open("w", encoding="utf-8") as log_file:
                proc = subprocess.Popen(cmd, cwd=SRC, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, encoding="utf-8", errors="replace", bufsize=1)
                assert proc.stdout is not None
                for line in proc.stdout:
                    log_file.write(line); log_file.flush()
                    self._append_log(job_id, line)
                code = proc.wait(timeout=30)
            if code != 0:
                raise RuntimeError(f"brain process exited with code {code}")
            with self._lock:
                job["status"] = "completed"
                job["finished_at"] = time.time()
                job["policy_path"] = str(policy_path.relative_to(ROOT)).replace("\\", "/")
                job["genome_dir"] = str(genome_dir.relative_to(ROOT)).replace("\\", "/")
        except Exception as exc:  # keep failures visible in the UI
            with self._lock:
                job["status"] = "failed"
                job["error"] = str(exc)
                job["finished_at"] = time.time()
                job["log"].append("ERROR: " + str(exc))

    def policy(self, kind: str) -> dict[str, Any]:
        self._settings(kind)
        path = POLICIES / f"{kind}.json"
        if not path.exists():
            raise FileNotFoundError(f"No {kind} policy exists yet. Train the brain first.")
        return json.loads(path.read_text(encoding="utf-8"))

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)


robotics_training = RoboticsTrainingManager()
