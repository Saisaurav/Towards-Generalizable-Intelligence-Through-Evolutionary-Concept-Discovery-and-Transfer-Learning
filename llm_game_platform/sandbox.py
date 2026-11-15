"""Parent-side client for the sandbox worker (one subprocess per environment instance)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

_WORKER = Path(__file__).with_name("sandbox_worker.py")


class SandboxError(RuntimeError):
    """The sandboxed environment failed. ``kind`` is contract|exception|timeout|memory|crash|load."""

    def __init__(self, message: str, kind: str = "exception", frames: str = "") -> None:
        super().__init__(message)
        self.kind = kind
        self.frames = frames

    def describe(self) -> str:
        text = f"[{self.kind}] {self}"
        return text + (f"\ngenerated code frames:\n{self.frames}" if self.frames else "")


class SandboxProcess:
    def __init__(self, source_path: Path | str, mem_mb: int = 1024, cpu_seconds: int = 600,
                 call_timeout: float = 10.0) -> None:
        self.source_path = str(Path(source_path).resolve())
        self.call_timeout = call_timeout
        self._stderr: deque[str] = deque(maxlen=40)
        self._busy_since: float | None = None
        self._killed_for_timeout = False
        self._cwd = tempfile.TemporaryDirectory(prefix="env-sandbox-")
        # Scrubbed environment: no API keys or credentials leak into generated code.
        env = {"PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
        if os.name == "nt":
            env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")
        command = [sys.executable, "-I", str(_WORKER), self.source_path, "--mem-mb", str(mem_mb),
                   "--cpu-seconds", str(cpu_seconds), "--call-timeout", str(call_timeout)]
        self.proc = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=self._cwd.name, env=env,
        )
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        threading.Thread(target=self._watchdog, daemon=True).start()
        loaded = self._read_json()
        if not loaded.get("ok"):
            self.close()
            raise SandboxError(loaded.get("error", "environment failed to load"), "load", loaded.get("frames", ""))
        self.spec: dict[str, Any] = loaded["spec"]

    # -- plumbing -----------------------------------------------------------
    def _drain_stderr(self) -> None:
        for raw in self.proc.stderr:  # type: ignore[union-attr]
            self._stderr.append(raw.decode("utf-8", "replace").rstrip())

    def _watchdog(self) -> None:
        # Backstop for platforms without SIGALRM and for code that blocks inside C.
        while self.proc.poll() is None:
            time.sleep(0.2)
            started = self._busy_since
            if started is not None and time.monotonic() - started > self.call_timeout + 3.0:
                self._killed_for_timeout = True
                self.proc.kill()
                return

    def _readline(self) -> str:
        line = self.proc.stdout.readline()  # type: ignore[union-attr]
        if not line:
            self.proc.wait()
            tail = "\n".join(list(self._stderr)[-8:])
            if self._killed_for_timeout:
                raise SandboxError("environment call timed out and the worker was killed", "timeout")
            code = self.proc.returncode
            hint = " (memory limit?)" if "MemoryError" in tail else ""
            raise SandboxError(f"sandbox worker exited unexpectedly (code {code}){hint}\n{tail}", "crash")
        return line.decode("utf-8").rstrip("\r\n")

    def _read_json(self) -> dict[str, Any]:
        return json.loads(self._readline())

    def _send(self, line: str) -> None:
        try:
            self.proc.stdin.write(line.encode("utf-8") + b"\n")  # type: ignore[union-attr]
            self.proc.stdin.flush()  # type: ignore[union-attr]
        except (BrokenPipeError, OSError):
            self._readline()  # raises with diagnostics
            raise SandboxError("sandbox worker closed its input", "crash")

    # -- API -------------------------------------------------------------------
    def request(self, op: str, **fields: Any) -> dict[str, Any]:
        self._busy_since = time.monotonic()
        try:
            self._send(json.dumps({"op": op, **fields}))
            response = self._read_json()
        finally:
            self._busy_since = None
        if not response.get("ok"):
            raise SandboxError(response.get("error", "unknown error"), response.get("kind", "exception"), response.get("frames", ""))
        return response

    def raw_line(self, line: str) -> str:
        """Brain-mode round trip used by the trainer relay (no JSON)."""
        self._busy_since = time.monotonic()
        try:
            self._send(line)
            return self._readline()
        finally:
            self._busy_since = None

    @property
    def stderr_tail(self) -> str:
        return "\n".join(self._stderr)

    def close(self) -> None:
        try:
            if self.proc.poll() is None:
                self.proc.kill()
            self.proc.wait(timeout=5)
        except Exception:
            pass
        for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            try:
                stream.close()  # type: ignore[union-attr]
            except Exception:
                pass
        self._cwd.cleanup()

    def __enter__(self) -> "SandboxProcess":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
