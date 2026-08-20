"""Tests for experiments/train_dynamic.py's env-construction helpers (T3.1, T3.2).

Only exercises environment wrapping/reset -- no SB3 model construction or actual
training (that stays untested-by-CI, matching this codebase's existing convention of
not unit-testing full training runs; ``experiments/run_ablations.py``/
``hpo_hetero.py`` are exercised the same way, by hand, not in the pytest suite).
"""

from __future__ import annotations

import numpy as np
import pytest

from envs.disruptions import DisruptionConfig
from envs.dynamic_jssp_env import DynamicJSSPEnv
from experiments.train_dynamic import (
    DomainRandomizedRegimeWrapper,
    frozen_regimes,
    make_dynamic_env_factory,
)


class TestDomainRandomizedRegimeWrapperNominalFraction:
    def test_zero_fraction_never_samples_nominal(self) -> None:
        env = DynamicJSSPEnv("ft06", DisruptionConfig())
        wrapped = DomainRandomizedRegimeWrapper(env, frozen_regimes(), seed=0, nominal_episode_fraction=0.0)
        for _ in range(30):
            wrapped.reset()
            assert wrapped.last_was_nominal is False
            assert wrapped.last_regime_index is not None

    def test_fraction_one_always_samples_nominal(self) -> None:
        env = DynamicJSSPEnv("ft06", DisruptionConfig())
        wrapped = DomainRandomizedRegimeWrapper(env, frozen_regimes(), seed=0, nominal_episode_fraction=1.0)
        for _ in range(10):
            wrapped.reset()
            assert wrapped.last_was_nominal is True
            assert wrapped.last_regime_index is None
            assert wrapped.env.unwrapped.config == DisruptionConfig()

    def test_effective_proportion_matches_configured_fraction(self) -> None:
        """T3.2's own explicit requirement: a test that verifies the EFFECTIVE
        proportion of nominal episodes, not just that the code path exists."""
        env = DynamicJSSPEnv("ft06", DisruptionConfig())
        wrapped = DomainRandomizedRegimeWrapper(env, frozen_regimes(), seed=42, nominal_episode_fraction=0.2)
        n = 2000
        n_nominal = 0
        for _ in range(n):
            wrapped.reset()
            if wrapped.last_was_nominal:
                n_nominal += 1
        observed = n_nominal / n
        # Binomial std at p=0.2, n=2000 is ~0.009 -- 3 std is a very safe tolerance.
        assert observed == pytest.approx(0.2, abs=0.03)

    def test_rejects_out_of_range_fraction(self) -> None:
        env = DynamicJSSPEnv("ft06", DisruptionConfig())
        with pytest.raises(ValueError):
            DomainRandomizedRegimeWrapper(env, frozen_regimes(), seed=0, nominal_episode_fraction=1.5)

    def test_default_fraction_is_zero_no_op(self) -> None:
        """The class's own default (as opposed to the CLI's 0.2 default) must stay 0.0
        -- Regla 5: existing direct callers of this class (if any) get the pre-T3.2
        behavior unless they opt in."""
        env = DynamicJSSPEnv("ft06", DisruptionConfig())
        wrapped = DomainRandomizedRegimeWrapper(env, frozen_regimes(), seed=0)
        for _ in range(20):
            wrapped.reset()
            assert wrapped.last_was_nominal is False


class TestMakeDynamicEnvFactoryTrainingRegime:
    def test_domain_randomized_is_default_and_unchanged(self) -> None:
        factory = make_dynamic_env_factory("ft06", kappa=0.0, seed=0)
        env = factory()
        assert isinstance(env.unwrapped, DynamicJSSPEnv)
        # Wrapper chain includes DomainRandomizedRegimeWrapper somewhere.
        wrapper_types = []
        w = env
        while hasattr(w, "env"):
            wrapper_types.append(type(w).__name__)
            w = w.env
        assert "DomainRandomizedRegimeWrapper" in wrapper_types

    def test_none_regime_has_no_domain_randomization_wrapper(self) -> None:
        factory = make_dynamic_env_factory("ft06", kappa=0.5, seed=0, training_regime="none")
        env = factory()
        wrapper_types = []
        w = env
        while hasattr(w, "env"):
            wrapper_types.append(type(w).__name__)
            w = w.env
        assert "DomainRandomizedRegimeWrapper" not in wrapper_types
        assert env.unwrapped.config == DisruptionConfig()

    def test_unknown_training_regime_raises(self) -> None:
        factory = make_dynamic_env_factory("ft06", kappa=0.0, seed=0, training_regime="bogus")
        with pytest.raises(ValueError, match="unknown training_regime"):
            factory()
