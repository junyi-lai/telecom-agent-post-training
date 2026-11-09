import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from telecom_post_training.deepseek_client import (
    _CURRENT_SIMULATOR,
    FALLBACK_CHANNEL,
    PRIMARY_CHANNEL,
    DeepSeekClientConfig,
    DeepSeekFallbackFusedError,
    DeepSeekRequestGate,
    DeepSeekTotalDeadlineExceeded,
    _retry_after_seconds,
    _retry_delay,
)
from telecom_post_training.settings import validate_deepseek_base_url


class RateLimitError(Exception):
    status_code = 429


class ServerError(Exception):
    status_code = 500


class AuthError(Exception):
    status_code = 401


class BadRequestError(Exception):
    status_code = 400


@pytest.mark.parametrize("alias", ["", "gateway-deployment"])
def test_request_alias_preserves_canonical_model(
    monkeypatch: pytest.MonkeyPatch, alias: str,
) -> None:
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("DEEPSEEK_MODEL", "canonical-model")
    monkeypatch.setenv("DEEPSEEK_REQUEST_MODEL", alias)
    monkeypatch.delenv("DEEPSEEK_FALLBACK_BASE_URL", raising=False)
    monkeypatch.delenv("DEEPSEEK_FALLBACK_API_KEY", raising=False)
    assert DeepSeekClientConfig.from_env().model == (alias or "canonical-model")
    assert os.environ["DEEPSEEK_MODEL"] == "canonical-model"


def _gate_config(tmp_path: Path, **overrides: object) -> DeepSeekClientConfig:
    values: dict[str, object] = {
        "primary_api_key": "test-primary-key",
        "gate_dir": tmp_path / "slots",
    }
    values.update(overrides)
    return DeepSeekClientConfig(**values)


def test_gate_limits_process_concurrency(tmp_path: Path) -> None:
    gate = DeepSeekRequestGate(
        _gate_config(
            tmp_path,
            max_concurrency=2,
            global_max_concurrency=2,
            telemetry_path=tmp_path / "telemetry.jsonl",
        )
    )
    active = 0
    peak = 0
    lock = threading.Lock()

    def request(**_: object) -> int:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return 1

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: gate.invoke(request), range(16)))

    snapshot = gate.snapshot()
    assert results == [1] * 16
    assert peak <= 2
    assert snapshot["max_active_requests"] == 2
    assert snapshot["active_requests"] == 0
    assert snapshot["successful_calls"] == 16
    assert snapshot["gate_scope"] == "process"


def test_gate_retries_and_records_rate_limit_and_timeout(tmp_path: Path) -> None:
    telemetry = tmp_path / "telemetry.jsonl"
    gate = DeepSeekRequestGate(
        _gate_config(
            tmp_path,
            max_concurrency=2,
            global_max_concurrency=2,
            max_retries=2,
            retry_base_seconds=0,
            telemetry_path=telemetry,
        )
    )
    attempts = 0

    def flaky(**_: object) -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RateLimitError("too many requests")
        if attempts == 2:
            raise TimeoutError("request timed out")
        return "ok"

    assert gate.invoke(flaky) == "ok"
    snapshot = gate.snapshot()
    assert snapshot["rate_limit_events"] == 1
    assert snapshot["timeout_events"] == 1
    assert snapshot["retries"] == 2
    assert snapshot["successful_calls"] == 1
    events = [
        json.loads(line) for line in telemetry.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events] == [
        "retry",
        "retry",
        "success",
    ]
    assert all(
        event["delay_seconds"] == 0 for event in events if event["event"] == "retry"
    )
    assert all(event["channel"] == PRIMARY_CHANNEL for event in events)


def test_gate_reduces_and_ramps_adaptive_concurrency(
    tmp_path: Path,
) -> None:
    gate = DeepSeekRequestGate(
        _gate_config(
            tmp_path,
            max_concurrency=4,
            min_concurrency=1,
            global_max_concurrency=4,
            adaptive_concurrency=True,
            adaptive_cooldown_seconds=0,
            successes_to_ramp=2,
            max_retries=1,
            retry_base_seconds=0,
            telemetry_path=tmp_path / "telemetry.jsonl",
        )
    )
    attempts = 0

    def flaky(**_: object) -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ServerError("temporary upstream failure")
        return "ok"

    assert gate.invoke(flaky) == "ok"
    assert gate.snapshot()["concurrency_limit"] == 3
    assert gate.invoke(lambda **_: "ok") == "ok"
    assert gate.snapshot()["concurrency_limit"] == 4


def test_gate_primary_success_prefers_primary(tmp_path: Path) -> None:
    gate = DeepSeekRequestGate(
        _gate_config(
            tmp_path,
            fallback_base_url="https://fallback.example/v1",
            fallback_api_key="fallback-secret",
            fallback_max_concurrency=2,
            fallback_min_concurrency=1,
            global_max_concurrency=2,
        )
    )

    assert gate.invoke(lambda **_: gate.current_channel()) == PRIMARY_CHANNEL
    assert gate.snapshot()["fallbacks"] == 0


def test_gate_uses_five_primary_retries_before_fallback(tmp_path: Path) -> None:
    telemetry = tmp_path / "telemetry.jsonl"
    gate = DeepSeekRequestGate(
        _gate_config(
            tmp_path,
            fallback_base_url="https://fallback.example/v1",
            fallback_api_key="fallback-secret",
            fallback_max_concurrency=2,
            fallback_min_concurrency=1,
            global_max_concurrency=2,
            max_retries=5,
            retry_base_seconds=0,
            telemetry_path=telemetry,
        )
    )
    calls: list[str] = []

    def routed(**_: object) -> str:
        channel = gate.current_channel()
        calls.append(channel)
        if channel == PRIMARY_CHANNEL:
            raise ServerError("temporary")
        return "ok"

    assert gate.invoke(routed) == "ok"
    assert calls == [PRIMARY_CHANNEL] * 6 + [FALLBACK_CHANNEL]
    snapshot = gate.snapshot()
    assert snapshot["retries"] == 5
    assert snapshot["fallbacks"] == 1
    assert snapshot["fallback_successes"] == 1
    events = [
        json.loads(line) for line in telemetry.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events] == [
        "retry",
        "retry",
        "retry",
        "retry",
        "retry",
        "fallback",
        "success",
    ]
    assert "fallback-secret" not in telemetry.read_text(encoding="utf-8")
    assert "test-primary-key" not in telemetry.read_text(encoding="utf-8")


def test_gate_pins_fallback_for_remaining_trajectory(tmp_path: Path) -> None:
    gate = DeepSeekRequestGate(
        _gate_config(
            tmp_path,
            fallback_base_url="https://fallback.example/v1",
            fallback_api_key="fallback-secret",
            fallback_max_concurrency=2,
            fallback_min_concurrency=1,
            global_max_concurrency=2,
            max_retries=1,
            retry_base_seconds=0,
        )
    )
    simulator = SimpleNamespace()
    calls: list[str] = []

    def first_call(**_: object) -> str:
        channel = gate.current_channel()
        calls.append(channel)
        if channel == PRIMARY_CHANNEL:
            raise ServerError("temporary")
        return "ok"

    token = _CURRENT_SIMULATOR.set(simulator)
    try:
        assert gate.invoke(first_call) == "ok"
        assert simulator._deepseek_channel_pin == FALLBACK_CHANNEL
        assert (
            gate.invoke(
                lambda **_: (
                    calls.append(gate.current_channel()) or gate.current_channel()
                )
            )
            == FALLBACK_CHANNEL
        )
    finally:
        _CURRENT_SIMULATOR.reset(token)
    assert calls == [
        PRIMARY_CHANNEL,
        PRIMARY_CHANNEL,
        FALLBACK_CHANNEL,
        FALLBACK_CHANNEL,
    ]


def test_gate_fuses_fallback_on_auth_failure(tmp_path: Path) -> None:
    fuse = tmp_path / "fallback.fused"
    gate = DeepSeekRequestGate(
        _gate_config(
            tmp_path,
            fallback_base_url="https://fallback.example/v1",
            fallback_api_key="fallback-secret",
            fallback_max_concurrency=2,
            fallback_min_concurrency=1,
            global_max_concurrency=2,
            fallback_fuse_path=fuse,
            max_retries=1,
            retry_base_seconds=0,
        )
    )
    fallback_calls = 0

    def routed(**_: object) -> str:
        nonlocal fallback_calls
        if gate.current_channel() == PRIMARY_CHANNEL:
            raise ServerError("temporary")
        fallback_calls += 1
        raise AuthError("invalid official key")

    with pytest.raises(AuthError):
        gate.invoke(routed)
    assert fuse.exists()
    assert gate.fallback_fused() is True
    second_gate = DeepSeekRequestGate(
        _gate_config(
            tmp_path,
            fallback_base_url="https://fallback.example/v1",
            fallback_api_key="fallback-secret",
            fallback_max_concurrency=2,
            fallback_min_concurrency=1,
            global_max_concurrency=2,
            fallback_fuse_path=fuse,
        )
    )
    assert second_gate.fallback_fused() is True

    with pytest.raises(DeepSeekFallbackFusedError):
        gate.invoke(routed)
    assert fallback_calls == 1


def test_cross_process_gate_caps_aggregate_concurrency(tmp_path: Path) -> None:
    common = {
        "max_concurrency": 2,
        "min_concurrency": 1,
        "global_max_concurrency": 1,
        "gate_dir": tmp_path / "shared-slots",
        "primary_api_key": "test-primary-key",
    }
    gates = [DeepSeekRequestGate(DeepSeekClientConfig(**common)) for _ in range(2)]
    active = 0
    peak = 0
    lock = threading.Lock()

    def request(gate: DeepSeekRequestGate, **_: object) -> int:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.04)
        with lock:
            active -= 1
        return 1

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(gates[index % 2].invoke, request, gates[index % 2])
            for index in range(8)
        ]
        assert [future.result() for future in futures] == [1] * 8
    assert peak == 1


def test_retry_delay_uses_bounded_jitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "telecom_post_training.deepseek_client.random.uniform",
        lambda low, high: high / 2,
    )
    assert (
        _retry_delay(
            retry_base_seconds=1,
            retry_jitter_ratio=0.5,
            attempt=1,
        )
        == 2.5
    )
    assert (
        _retry_delay(
            retry_base_seconds=10,
            retry_jitter_ratio=0.25,
            attempt=4,
            max_seconds=1,
        )
        == 1
    )


def test_retry_after_header_is_capped() -> None:
    error = RateLimitError("too many requests")
    error.response = type(
        "Response",
        (),
        {"headers": {"retry-after": "12.5"}},
    )()
    assert _retry_after_seconds(error, max_seconds=1) == 1

    error.response.headers["retry-after"] = "999"
    assert _retry_after_seconds(error, max_seconds=1) == 1


def test_non_retryable_400_is_not_retried(tmp_path: Path) -> None:
    gate = DeepSeekRequestGate(
        _gate_config(
            tmp_path,
            max_retries=5,
            retry_base_seconds=0,
        )
    )
    calls = 0

    def bad_request(**_: object) -> str:
        nonlocal calls
        calls += 1
        raise BadRequestError("invalid request")

    with pytest.raises(BadRequestError):
        gate.invoke(bad_request)
    assert calls == 1
    assert gate.snapshot()["retries"] == 0
    assert gate.snapshot()["failures"] == 1


def test_gate_stops_at_total_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = SimpleNamespace(now=0.0)
    observed_timeouts: list[float] = []
    monkeypatch.setattr(
        "telecom_post_training.deepseek_client.time.monotonic",
        lambda: clock.now,
    )
    gate = DeepSeekRequestGate(
        _gate_config(
            tmp_path,
            max_retries=5,
            retry_base_seconds=0,
            retry_jitter_ratio=0,
            request_timeout_seconds=20,
            total_request_deadline_seconds=60,
        )
    )

    def slow_server(**kwargs: object) -> str:
        observed_timeouts.append(float(kwargs["timeout"]))
        clock.now += 11
        raise ServerError("temporary")

    with pytest.raises(DeepSeekTotalDeadlineExceeded):
        gate.invoke(slow_server)
    assert len(observed_timeouts) == 6
    assert observed_timeouts[-1] == 5
    assert gate.snapshot()["attempts"] == 6
    assert gate.snapshot()["fallbacks"] == 0


def test_backoff_cannot_run_past_total_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = SimpleNamespace(now=0.0)

    def fake_sleep(seconds: float) -> None:
        clock.now += seconds

    monkeypatch.setattr(
        "telecom_post_training.deepseek_client.time.monotonic",
        lambda: clock.now,
    )
    monkeypatch.setattr(
        "telecom_post_training.deepseek_client.time.sleep",
        fake_sleep,
    )
    gate = DeepSeekRequestGate(
        _gate_config(
            tmp_path,
            max_retries=5,
            retry_base_seconds=10,
            retry_jitter_ratio=0,
            retry_max_seconds=1,
            total_request_deadline_seconds=0.5,
        )
    )
    calls = 0

    def failing(**_: object) -> str:
        nonlocal calls
        calls += 1
        raise ServerError("temporary")

    with pytest.raises(DeepSeekTotalDeadlineExceeded):
        gate.invoke(failing)
    assert calls == 1
    assert gate.snapshot()["retries"] == 1


def test_fallback_uses_remaining_total_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(
        "telecom_post_training.deepseek_client.time.monotonic",
        lambda: clock.now,
    )
    gate = DeepSeekRequestGate(
        _gate_config(
            tmp_path,
            fallback_base_url="https://fallback.example/v1",
            fallback_api_key="fallback-secret",
            fallback_max_concurrency=2,
            fallback_min_concurrency=1,
            max_retries=0,
            request_timeout_seconds=20,
            total_request_deadline_seconds=60,
        )
    )
    fallback_timeout: list[float] = []

    def routed(**kwargs: object) -> str:
        if gate.current_channel() == PRIMARY_CHANNEL:
            clock.now += 59
            raise ServerError("temporary")
        fallback_timeout.append(float(kwargs["timeout"]))
        return "ok"

    assert gate.invoke(routed) == "ok"
    assert fallback_timeout == [1]


def test_deadline_is_not_cleared_when_expired(tmp_path: Path) -> None:
    fuse = tmp_path / "fallback.fused"
    fuse.write_text("{}\n", encoding="utf-8")
    old = time.time() - 7200
    Path(fuse).touch()
    os.utime(fuse, (old, old))
    gate = DeepSeekRequestGate(
        _gate_config(
            tmp_path,
            fallback_base_url="https://fallback.example/v1",
            fallback_api_key="fallback-secret",
            fallback_max_concurrency=2,
            fallback_min_concurrency=1,
            fallback_fuse_path=fuse,
            fallback_fuse_seconds=3600,
        )
    )
    assert gate.fallback_fused() is False
    assert fuse.exists()


def test_deepseek_url_requires_explicit_public_http_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert validate_deepseek_base_url("https://gateway.example.com/v1")
    assert validate_deepseek_base_url("http://127.0.0.1:8080/v1")
    assert validate_deepseek_base_url("http://10.0.0.8:8080/v1")
    monkeypatch.delenv("ALLOW_INSECURE_DEEPSEEK_HTTP", raising=False)
    with pytest.raises(ValueError, match="must use HTTPS"):
        validate_deepseek_base_url("http://8.8.8.8:8080/v1")
    monkeypatch.setenv("ALLOW_INSECURE_DEEPSEEK_HTTP", "1")
    assert validate_deepseek_base_url("http://8.8.8.8:8080/v1")
    with pytest.raises(ValueError, match="must use HTTPS"):
        validate_deepseek_base_url(
            "http://8.8.8.8:8080/v1",
            allow_insecure_http=False,
        )


def test_config_repr_and_gate_snapshot_do_not_expose_keys(
    tmp_path: Path,
) -> None:
    primary_secret = "primary-super-secret"
    fallback_secret = "fallback-super-secret"
    config = DeepSeekClientConfig(
        base_url="https://primary.example/v1",
        primary_api_key=primary_secret,
        fallback_base_url="https://fallback.example/v1",
        fallback_api_key=fallback_secret,
        gate_dir=tmp_path / "slots",
    )
    rendered = repr(config) + json.dumps(
        DeepSeekRequestGate(config).snapshot(),
        sort_keys=True,
    )
    assert primary_secret not in rendered
    assert fallback_secret not in rendered


def test_config_from_env_builds_primary_and_fallback_routes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "http://10.0.0.2:8080/v1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "primary-secret")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-flash")
    monkeypatch.setenv(
        "DEEPSEEK_FALLBACK_BASE_URL",
        "https://api.deepseek.com/v1",
    )
    monkeypatch.setenv("DEEPSEEK_FALLBACK_API_KEY", "fallback-secret")
    monkeypatch.setenv("DEEPSEEK_FALLBACK_MODEL", "deepseek-flash")
    monkeypatch.setenv("DEEPSEEK_MAX_CONCURRENCY", "4")
    monkeypatch.setenv("DEEPSEEK_MIN_CONCURRENCY", "1")
    monkeypatch.setenv("DEEPSEEK_GLOBAL_MAX_CONCURRENCY", "64")
    monkeypatch.setenv("DEEPSEEK_FALLBACK_MAX_CONCURRENCY", "4")
    monkeypatch.setenv("DEEPSEEK_FALLBACK_MIN_CONCURRENCY", "2")
    monkeypatch.setenv("DEEPSEEK_FALLBACK_GLOBAL_MAX_CONCURRENCY", "4")
    monkeypatch.setenv("DEEPSEEK_REQUEST_TIMEOUT_SECONDS", "20")
    monkeypatch.setenv("DEEPSEEK_MAX_RETRIES", "5")
    monkeypatch.setenv("DEEPSEEK_RETRY_BASE_SECONDS", "0.1")
    monkeypatch.setenv("DEEPSEEK_RETRY_JITTER_RATIO", "0.25")
    monkeypatch.setenv("DEEPSEEK_RETRY_MAX_SECONDS", "1")
    monkeypatch.setenv("DEEPSEEK_TOTAL_REQUEST_DEADLINE_SECONDS", "60")
    monkeypatch.setenv("AGENT_WORKERS", "16")
    monkeypatch.setenv("DEEPSEEK_GATE_DIR", str(tmp_path / "slots"))
    config = DeepSeekClientConfig.from_env()

    assert config.fallback_enabled is True
    assert config.max_concurrency == 4
    assert config.min_concurrency == 1
    assert config.global_max_concurrency == 64
    assert config.fallback_global_max_concurrency == 4
    assert config.request_timeout_seconds == 20
    assert config.max_retries == 5
    assert config.retry_base_seconds == 0.1
    assert config.retry_jitter_ratio == 0.25
    assert config.retry_max_seconds == 1
    assert config.total_request_deadline_seconds == 60
    assert "primary-secret" not in repr(config)
    assert "fallback-secret" not in repr(config)
