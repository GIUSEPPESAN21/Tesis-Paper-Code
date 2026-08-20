"""Statistical validation of the three disruption generators, each in isolation.

Every generator is exercised directly against :class:`~envs.instance.JSSPInstance`
without touching :class:`~envs.dynamic_jssp_env.DynamicJSSPEnv` at all --- the whole
point of the generators living in their own module. Tolerances follow
``DynamicEnv_TechSpec.md`` §8: ``N >= 1000`` draws per assertion, a fixed relative
tolerance (5%, 10% for the noisier variance check) with a fixed master seed rather than a
p-value-threshold hypothesis test (to keep CI deterministic), except for one dedicated KS
"shape" test per distribution.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from envs.disruptions import (
    ArrivalProcess,
    BreakdownGenerator,
    DisruptionConfig,
    ProcessingTimeSampler,
    spawn_disruption_rngs,
)
from envs.loader import load_by_name

N = 5000
RTOL = 0.05


@pytest.fixture(scope="module")
def instance():
    return load_by_name("ft06")


class TestBreakdownGeneratorMeans:
    @pytest.mark.parametrize("availability", [0.95, 0.90, 0.80])
    def test_breakdown_generator_mean_ttf_matches_target(self, instance, availability: float) -> None:
        config = DisruptionConfig(target_availability=availability, seed=0)
        rng = np.random.default_rng(1)
        gen = BreakdownGenerator(instance, config, rng)

        samples = np.array([gen.draw_ttf(0) for _ in range(N)])
        empirical_mean = samples.mean()
        assert empirical_mean == pytest.approx(gen.mtbf[0], rel=RTOL)

    def test_breakdown_generator_mean_ttr_matches_target(self, instance) -> None:
        config = DisruptionConfig(target_availability=0.9, mttr_multiplier=2.0, seed=0)
        rng = np.random.default_rng(2)
        gen = BreakdownGenerator(instance, config, rng)

        samples = np.array([gen.draw_ttr() for _ in range(N)])
        assert samples.mean() == pytest.approx(gen.mttr, rel=RTOL)

    def test_mean_ttr_independent_of_availability(self, instance) -> None:
        """MTTR is fixed by mttr_multiplier alone, not by the target availability."""
        rng_low = np.random.default_rng(3)
        rng_high = np.random.default_rng(3)
        gen_low = BreakdownGenerator(instance, DisruptionConfig(target_availability=0.6, seed=0), rng_low)
        gen_high = BreakdownGenerator(instance, DisruptionConfig(target_availability=0.98, seed=0), rng_high)
        assert gen_low.mttr == pytest.approx(gen_high.mttr)


class TestBreakdownGeneratorShape:
    def test_breakdown_generator_ttf_shape_ks(self, instance) -> None:
        config = DisruptionConfig(target_availability=0.85, seed=0)
        rng = np.random.default_rng(4)
        gen = BreakdownGenerator(instance, config, rng)

        samples = np.array([gen.draw_ttf(0) for _ in range(N)])
        result = stats.kstest(samples, "expon", args=(0, 1.0 / gen.lambda_k[0]))
        assert result.pvalue > 0.01, f"KS test rejects Exp({gen.lambda_k[0]:g}): p={result.pvalue:g}"


class TestBreakdownGeneratorAvailabilityRenewal:
    @pytest.mark.parametrize("availability", [0.95, 0.90, 0.80])
    def test_breakdown_generator_availability_recovers_target(self, instance, availability: float) -> None:
        config = DisruptionConfig(target_availability=availability, seed=0)
        rng = np.random.default_rng(5)
        gen = BreakdownGenerator(instance, config, rng)

        total_up = 0.0
        total_down = 0.0
        for _ in range(N):
            total_up += gen.draw_ttf(0)
            total_down += gen.draw_ttr()

        empirical_availability = total_up / (total_up + total_down)
        assert empirical_availability == pytest.approx(availability, rel=RTOL)

    def test_availability_one_disables_breakdowns(self, instance) -> None:
        config = DisruptionConfig(target_availability=1.0, seed=0)
        rng = np.random.default_rng(6)
        gen = BreakdownGenerator(instance, config, rng)
        assert not gen.is_enabled(0)
        assert gen.draw_ttf(0) == float("inf")


class TestProcessingTimeSampler:
    @pytest.mark.parametrize("sigma", [0.1, 0.25, 0.5])
    def test_processing_time_sampler_mean_is_one(self, sigma: float) -> None:
        rng = np.random.default_rng(7)
        sampler = ProcessingTimeSampler(sigma, rng)
        samples = np.array([sampler.draw_multiplier() for _ in range(N)])
        assert samples.mean() == pytest.approx(1.0, rel=RTOL, abs=0.02)

    def test_processing_time_sampler_variance_scales_with_s(self) -> None:
        """Variance increases monotonically with s and matches the closed form.

        For X ~ Lognormal(mu, s^2) with mu = -s^2/2 (so E[X] = 1 by construction),
        Var(X) = (exp(s^2) - 1) * exp(2*mu + s^2) = (exp(s^2) - 1) * exp(0) = exp(s^2) - 1.
        """
        sigmas = [0.1, 0.25, 0.5]
        variances = []
        for sigma in sigmas:
            rng = np.random.default_rng(8)
            sampler = ProcessingTimeSampler(sigma, rng)
            samples = np.array([sampler.draw_multiplier() for _ in range(N)])
            empirical_var = samples.var()
            theoretical_var = np.exp(sigma**2) - 1.0
            assert empirical_var == pytest.approx(theoretical_var, rel=0.10)
            variances.append(empirical_var)

        assert variances[0] < variances[1] < variances[2]

    def test_disabled_sampler_is_deterministic(self) -> None:
        rng = np.random.default_rng(9)
        sampler = ProcessingTimeSampler(0.0, rng)
        assert sampler.draw_multiplier() == 1.0
        assert sampler.effective_duration(7.0) == 7.0


class TestArrivalProcess:
    @pytest.mark.parametrize("rho", [0.70, 0.85, 0.95])
    def test_arrival_process_rate_matches_target_utilization(self, instance, rho: float) -> None:
        config = DisruptionConfig(
            target_utilization=rho,
            arrival_cutoff_time=1e12,  # effectively disable the cutoff for this rate test
            max_dynamic_jobs=100_000,
            seed=0,
        )
        rng = np.random.default_rng(10)
        process = ArrivalProcess(instance, config, rng)

        t = 0.0
        gaps = []
        for _ in range(N):
            next_t = process.next_arrival_time(t)
            assert next_t is not None
            gaps.append(next_t - t)
            t = next_t
            process.generate_job()

        empirical_lambda_a = 1.0 / np.mean(gaps)
        assert empirical_lambda_a == pytest.approx(process.lambda_a, rel=RTOL)

    def test_arrival_process_stops_at_cutoff(self, instance) -> None:
        config = DisruptionConfig(
            target_utilization=0.9, arrival_cutoff_time=50.0, max_dynamic_jobs=1000, seed=0
        )
        rng = np.random.default_rng(11)
        process = ArrivalProcess(instance, config, rng)

        t = 0.0
        arrival_times = []
        while True:
            next_t = process.next_arrival_time(t)
            if next_t is None:
                break
            assert next_t <= config.arrival_cutoff_time
            arrival_times.append(next_t)
            t = next_t
            process.generate_job()

        assert all(at <= 50.0 for at in arrival_times)
        assert process.num_generated == len(arrival_times)

    def test_arrival_process_stops_at_max_dynamic_jobs(self, instance) -> None:
        config = DisruptionConfig(
            target_utilization=0.9, arrival_cutoff_time=1e12, max_dynamic_jobs=5, seed=0
        )
        rng = np.random.default_rng(12)
        process = ArrivalProcess(instance, config, rng)

        t = 0.0
        count = 0
        while True:
            next_t = process.next_arrival_time(t)
            if next_t is None:
                break
            process.generate_job()
            t = next_t
            count += 1
        assert count == 5
        assert process.num_generated == 5

    def test_disabled_arrival_process_never_arrives(self, instance) -> None:
        config = DisruptionConfig(target_utilization=None, seed=0)
        rng = np.random.default_rng(13)
        process = ArrivalProcess(instance, config, rng)
        assert process.next_arrival_time(0.0) is None

    def test_generate_job_preserves_rectangularity_and_machine_uniqueness(self, instance) -> None:
        config = DisruptionConfig(target_utilization=0.8, seed=0)
        rng = np.random.default_rng(14)
        process = ArrivalProcess(instance, config, rng)
        for _ in range(20):
            job = process.generate_job()
            assert job["machines"].shape == (instance.ops_per_job,)
            assert job["durations"].shape == (instance.ops_per_job,)
            assert len(set(job["machines"].tolist())) == instance.ops_per_job
            assert set(job["durations"].tolist()) <= set(instance.flat_durations.tolist())


class TestIndependentRngStreams:
    def test_generators_independent_of_seed_spawn(self, instance) -> None:
        """Toggling processing_time_sigma must not perturb the breakdown draw sequence:
        each generator gets its own independent child stream from spawn(3), regardless
        of which other disruption axes are enabled.
        """
        seed = 42

        breakdown_rng_a, _, _ = spawn_disruption_rngs(seed)
        config_a = DisruptionConfig(target_availability=0.9, processing_time_sigma=0.0, seed=seed)
        gen_a = BreakdownGenerator(instance, config_a, breakdown_rng_a)
        ttf_sequence_a = [gen_a.draw_ttf(0) for _ in range(50)]

        breakdown_rng_b, _, _ = spawn_disruption_rngs(seed)
        config_b = DisruptionConfig(target_availability=0.9, processing_time_sigma=0.5, seed=seed)
        gen_b = BreakdownGenerator(instance, config_b, breakdown_rng_b)
        ttf_sequence_b = [gen_b.draw_ttf(0) for _ in range(50)]

        assert ttf_sequence_a == ttf_sequence_b

    def test_three_spawned_streams_are_mutually_independent(self) -> None:
        rng_a, rng_b, rng_c = spawn_disruption_rngs(0)
        draws_a = rng_a.random(10)
        draws_b = rng_b.random(10)
        draws_c = rng_c.random(10)
        assert not np.allclose(draws_a, draws_b)
        assert not np.allclose(draws_a, draws_c)
        assert not np.allclose(draws_b, draws_c)

    def test_spawn_disruption_rngs_reproducible_for_same_seed(self) -> None:
        a1, b1, c1 = spawn_disruption_rngs(123)
        a2, b2, c2 = spawn_disruption_rngs(123)
        assert a1.random(5).tolist() == a2.random(5).tolist()
        assert b1.random(5).tolist() == b2.random(5).tolist()
        assert c1.random(5).tolist() == c2.random(5).tolist()
