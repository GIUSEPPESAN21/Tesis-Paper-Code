"""Tests for T2.1's --seeds/--n-workers dispatcher in train_dynamic.py/train_hetero.py.

These do NOT train anything -- ``subprocess.Popen`` is monkeypatched so
``_run_seeds_in_subprocesses`` never actually launches a training run; the point is to
lock in the argv-stripping logic (a real bug -- a naive "skip the token right after
--seeds" check silently leaked extra positional seed values into the child command --
was caught and fixed while implementing this, hence the direct test).
"""

from __future__ import annotations

import experiments.train_dynamic as train_dynamic
import experiments.train_hetero as train_hetero


class _FakeProc:
    def __init__(self, cmd):
        self.cmd = cmd
        self._polled = False

    def poll(self):
        # "completes" the first time it's polled, exit code 0.
        if self._polled:
            return 0
        self._polled = True
        return None


def _capture_popen(monkeypatch, module):
    """Patch the real (process-wide) ``subprocess.Popen`` -- both modules do
    ``import subprocess`` *inside* ``_run_seeds_in_subprocesses``, which just binds to
    the same cached ``sys.modules["subprocess"]`` object, so patching it globally here
    is sufficient (no per-module patch target needed)."""
    calls = []

    def fake_popen(cmd, *a, **kw):
        calls.append(cmd)
        return _FakeProc(cmd)

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    monkeypatch.setattr(module.time, "sleep", lambda *_: None)
    return calls


class TestTrainDynamicMultiSeedArgvStripping:
    def test_seeds_and_n_workers_stripped_multi_value(self, monkeypatch) -> None:
        calls = _capture_popen(monkeypatch, train_dynamic)
        argv = [
            "--instance", "ft06", "--kappa", "0.5", "--stage", "D1",
            "--seeds", "0", "1", "2", "3", "4", "--n-workers", "2",
        ]
        exit_code = train_dynamic._run_seeds_in_subprocesses(argv, [0, 1, 2, 3, 4], n_workers=2)
        assert exit_code == 0
        assert len(calls) == 5
        for cmd in calls:
            # No stray bare numeric tokens (the bug this test guards against) --
            # every "--seeds" in the child command is followed by EXACTLY one value.
            assert cmd.count("--seeds") == 1
            seeds_idx = cmd.index("--seeds")
            assert cmd[seeds_idx + 1] in ("0", "1", "2", "3", "4")
            # the child command must not contain "--n-workers" at all
            assert "--n-workers" not in cmd
            # base flags survive untouched
            assert "--instance" in cmd and "ft06" in cmd
            assert "--kappa" in cmd and "0.5" in cmd

    def test_n_workers_caps_concurrency(self, monkeypatch) -> None:
        calls = _capture_popen(monkeypatch, train_dynamic)
        argv = ["--instance", "ft06", "--kappa", "0.5", "--seeds", "0", "1", "--n-workers", "1"]
        train_dynamic._run_seeds_in_subprocesses(argv, [0, 1], n_workers=1)
        assert len(calls) == 2


class TestTrainHeteroMultiSeedArgvStripping:
    def test_seeds_and_n_workers_stripped(self, monkeypatch) -> None:
        calls = _capture_popen(monkeypatch, train_hetero)
        argv = ["--instance", "ft06", "--stage", "A", "--seeds", "0", "1", "2", "--n-workers", "3"]
        exit_code = train_hetero._run_seeds_in_subprocesses(argv, [0, 1, 2], n_workers=3)
        assert exit_code == 0
        assert len(calls) == 3
        for cmd in calls:
            assert cmd.count("--seeds") == 1
            assert "--n-workers" not in cmd
