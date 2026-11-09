from __future__ import annotations

import contextvars
import fcntl
import json
import os
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from telecom_post_training.settings import validate_deepseek_base_url

PRIMARY_CHANNEL = "primary"
FALLBACK_CHANNEL = "fallback"

_ACTIVE_CHANNEL: contextvars.ContextVar[str] = contextvars.ContextVar(
    "deepseek_active_channel",
    default=PRIMARY_CHANNEL,
)
_CURRENT_SIMULATOR: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "deepseek_current_simulator",
    default=None,
)


@dataclass(frozen=True)
class DeepSeekClientConfig:
    base_url: str = "https://api.deepseek.com/v1"
    primary_api_key: str = field(default="", repr=False)
    model: str = "deepseek-flash"
    fallback_base_url: str = ""
    fallback_api_key: str = field(default="", repr=False)
    fallback_model: str = "deepseek-flash"
    max_concurrency: int = 16
    min_concurrency: int = 1
    fallback_max_concurrency: int = 4
    fallback_min_concurrency: int = 2
    global_max_concurrency: int = 64
    fallback_global_max_concurrency: int = 4
    worker_processes: int = 1
    adaptive_concurrency: bool = True
    adaptive_cooldown_seconds: float = 2.0
    successes_to_ramp: int = 8
    request_timeout_seconds: float = 90.0
    max_retries: int = 3
    retry_base_seconds: float = 1.0
    retry_jitter_ratio: float = 0.35
    retry_max_seconds: float = 1.0
    total_request_deadline_seconds: float = 60.0
    fallback_fuse_seconds: float = 3600.0
    fallback_fuse_path: Path | None = None
    gate_dir: Path = Path("/tmp/telecom_post_training_deepseek_slots")
    telemetry_path: Path | None = None

    @property
    def fallback_enabled(self) -> bool:
        return bool(self.fallback_base_url and self.fallback_api_key)

    @classmethod
    def from_env(cls) -> "DeepSeekClientConfig":
        base_url = os.getenv(
            "DEEPSEEK_BASE_URL",
            "https://api.deepseek.com/v1",
        )
        validate_deepseek_base_url(base_url)
        fallback_base_url = os.getenv("DEEPSEEK_FALLBACK_BASE_URL", "").strip()
        fallback_api_key = os.getenv("DEEPSEEK_FALLBACK_API_KEY", "").strip()
        if bool(fallback_base_url) != bool(fallback_api_key):
            raise ValueError(
                "DEEPSEEK_FALLBACK_BASE_URL and "
                "DEEPSEEK_FALLBACK_API_KEY must be configured together"
            )
        if fallback_base_url:
            validate_deepseek_base_url(fallback_base_url)

        max_concurrency = int(os.getenv("DEEPSEEK_MAX_CONCURRENCY", "16"))
        worker_processes = max(1, int(os.getenv("AGENT_WORKERS", "1")))
        config = cls(
            base_url=base_url,
            primary_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            model=(
                os.getenv("DEEPSEEK_REQUEST_MODEL", "").strip()
                or os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
            ),
            fallback_base_url=fallback_base_url,
            fallback_api_key=fallback_api_key,
            fallback_model=os.getenv(
                "DEEPSEEK_FALLBACK_MODEL",
                "deepseek-flash",
            ),
            max_concurrency=max_concurrency,
            min_concurrency=int(
                os.getenv(
                    "DEEPSEEK_MIN_CONCURRENCY",
                    str(min(4, max_concurrency)),
                )
            ),
            fallback_max_concurrency=int(
                os.getenv("DEEPSEEK_FALLBACK_MAX_CONCURRENCY", "4")
            ),
            fallback_min_concurrency=int(
                os.getenv("DEEPSEEK_FALLBACK_MIN_CONCURRENCY", "2")
            ),
            global_max_concurrency=int(
                os.getenv(
                    "DEEPSEEK_GLOBAL_MAX_CONCURRENCY",
                    str(max_concurrency * worker_processes),
                )
            ),
            fallback_global_max_concurrency=int(
                os.getenv("DEEPSEEK_FALLBACK_GLOBAL_MAX_CONCURRENCY", "4")
            ),
            worker_processes=worker_processes,
            adaptive_concurrency=_env_flag(
                "DEEPSEEK_ADAPTIVE_CONCURRENCY",
                default=True,
            ),
            adaptive_cooldown_seconds=float(
                os.getenv("DEEPSEEK_ADAPTIVE_COOLDOWN_SECONDS", "2")
            ),
            successes_to_ramp=int(os.getenv("DEEPSEEK_SUCCESSES_TO_RAMP", "8")),
            request_timeout_seconds=float(
                os.getenv("DEEPSEEK_REQUEST_TIMEOUT_SECONDS", "90")
            ),
            max_retries=int(os.getenv("DEEPSEEK_MAX_RETRIES", "3")),
            retry_base_seconds=float(os.getenv("DEEPSEEK_RETRY_BASE_SECONDS", "1")),
            retry_jitter_ratio=float(os.getenv("DEEPSEEK_RETRY_JITTER_RATIO", "0.35")),
            retry_max_seconds=float(
                os.getenv("DEEPSEEK_RETRY_MAX_SECONDS", "1")
            ),
            total_request_deadline_seconds=float(
                os.getenv("DEEPSEEK_TOTAL_REQUEST_DEADLINE_SECONDS", "60")
            ),
            fallback_fuse_seconds=float(
                os.getenv("DEEPSEEK_FALLBACK_FUSE_SECONDS", "3600")
            ),
            fallback_fuse_path=Path(
                os.getenv(
                    "DEEPSEEK_FALLBACK_FUSE_PATH",
                    "/tmp/telecom_post_training_deepseek_fallback.fused",
                )
            ).expanduser(),
            gate_dir=Path(
                os.getenv(
                    "DEEPSEEK_GATE_DIR",
                    "/tmp/telecom_post_training_deepseek_slots",
                )
            ).expanduser(),
            telemetry_path=(
                Path(os.environ["DEEPSEEK_TELEMETRY_PATH"]).expanduser()
                if os.getenv("DEEPSEEK_TELEMETRY_PATH")
                else None
            ),
        )
        _validate_config(config)
        return config


def _validate_config(config: DeepSeekClientConfig) -> None:
    if not 1 <= config.max_concurrency <= 64:
        raise ValueError("DEEPSEEK_MAX_CONCURRENCY must be between 1 and 64")
    if not 1 <= config.min_concurrency <= config.max_concurrency:
        raise ValueError(
            "DEEPSEEK_MIN_CONCURRENCY must be between 1 and DEEPSEEK_MAX_CONCURRENCY"
        )
    if not 1 <= config.global_max_concurrency <= 256:
        raise ValueError("DEEPSEEK_GLOBAL_MAX_CONCURRENCY must be between 1 and 256")
    if config.fallback_enabled:
        if not 1 <= config.fallback_max_concurrency <= 64:
            raise ValueError(
                "DEEPSEEK_FALLBACK_MAX_CONCURRENCY must be between 1 and 64"
            )
        if not (
            1 <= config.fallback_min_concurrency <= config.fallback_max_concurrency
        ):
            raise ValueError(
                "DEEPSEEK_FALLBACK_MIN_CONCURRENCY must be between 1 and "
                "DEEPSEEK_FALLBACK_MAX_CONCURRENCY"
            )
        if not 1 <= config.fallback_global_max_concurrency <= 64:
            raise ValueError(
                "DEEPSEEK_FALLBACK_GLOBAL_MAX_CONCURRENCY must be between 1 and 64"
            )
    if config.worker_processes < 1:
        raise ValueError("AGENT_WORKERS must be positive")
    if config.adaptive_cooldown_seconds < 0:
        raise ValueError("DEEPSEEK_ADAPTIVE_COOLDOWN_SECONDS must be nonnegative")
    if config.successes_to_ramp < 1:
        raise ValueError("DEEPSEEK_SUCCESSES_TO_RAMP must be positive")
    if config.request_timeout_seconds <= 0:
        raise ValueError("DEEPSEEK_REQUEST_TIMEOUT_SECONDS must be positive")
    if not 0 <= config.max_retries <= 10:
        raise ValueError("DEEPSEEK_MAX_RETRIES must be between 0 and 10")
    if config.retry_base_seconds < 0:
        raise ValueError("DEEPSEEK_RETRY_BASE_SECONDS must be nonnegative")
    if not 0 <= config.retry_jitter_ratio <= 2:
        raise ValueError("DEEPSEEK_RETRY_JITTER_RATIO must be in [0, 2]")
    if config.retry_max_seconds < 0:
        raise ValueError("DEEPSEEK_RETRY_MAX_SECONDS must be nonnegative")
    if config.total_request_deadline_seconds <= 0:
        raise ValueError(
            "DEEPSEEK_TOTAL_REQUEST_DEADLINE_SECONDS must be positive"
        )
    if config.fallback_fuse_seconds < 0:
        raise ValueError("DEEPSEEK_FALLBACK_FUSE_SECONDS must be nonnegative")


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean flag")


def _status_code(error: Exception) -> int | None:
    status = getattr(error, "status_code", None)
    if status is None:
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _is_timeout(error: Exception) -> bool:
    return isinstance(error, TimeoutError) or "timeout" in type(error).__name__.lower()


def _retry_after_seconds(error: Exception, *, max_seconds: float = 30.0) -> float:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return 0.0
    value = headers.get("retry-after")
    if value is None:
        return 0.0
    try:
        return min(max(float(value), 0.0), max_seconds)
    except (TypeError, ValueError):
        return 0.0


def _is_retryable(error: Exception, status: int | None) -> bool:
    if status == 429 or (status is not None and status >= 500):
        return True
    if _is_timeout(error):
        return True
    return any(
        token in type(error).__name__.lower()
        for token in ("connection", "serviceunavailable", "ratelimit")
    )


def _is_fallback_fatal(error: Exception, status: int | None) -> bool:
    if status in {401, 402, 403}:
        return True
    message = str(error).lower()
    return any(
        token in message
        for token in (
            "insufficient balance",
            "insufficient_quota",
            "billing",
            "balance is not enough",
        )
    )


def _retry_delay(
    retry_base_seconds: float,
    retry_jitter_ratio: float,
    attempt: int,
    *,
    max_seconds: float = 30.0,
) -> float:
    base = min(retry_base_seconds * (2**attempt), max_seconds)
    jitter = random.uniform(0.0, base * retry_jitter_ratio)
    return min(base + jitter, max_seconds)


class DeepSeekFallbackFusedError(RuntimeError):
    """Raised when the official fallback channel is circuit-broken."""


class DeepSeekTotalDeadlineExceeded(TimeoutError):
    """Raised when the per-request total deadline is exhausted."""


class _FileSlotLimiter:
    """Cross-process concurrency slots backed by advisory file locks."""

    def __init__(self, root: Path, name: str, limit: int):
        self.root = root / f"{name}-{limit}"
        self.limit = limit
        self.root.mkdir(parents=True, exist_ok=True)

    def acquire(self) -> int | None:
        while True:
            for index in range(self.limit):
                path = self.root / f"slot-{index:03d}.lock"
                descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    os.close(descriptor)
                    continue
                return descriptor
            time.sleep(0.02)

    def release(self, descriptor: int | None) -> None:
        if descriptor is not None:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


@dataclass(frozen=True)
class _ChannelRoute:
    name: str
    base_url: str
    api_key: str = field(repr=False)
    model: str = "deepseek-flash"
    max_concurrency: int = 1
    min_concurrency: int = 1
    global_max_concurrency: int = 1


class DeepSeekRequestGate:
    def __init__(self, config: DeepSeekClientConfig):
        _validate_config(config)
        self.config = config
        self._condition = threading.Condition()
        self._limits = {
            PRIMARY_CHANNEL: config.max_concurrency,
            FALLBACK_CHANNEL: config.fallback_max_concurrency,
        }
        self._last_reduction = {
            PRIMARY_CHANNEL: float("-inf"),
            FALLBACK_CHANNEL: float("-inf"),
        }
        self._success_streak = {
            PRIMARY_CHANNEL: 0,
            FALLBACK_CHANNEL: 0,
        }
        self._active_by_channel = {
            PRIMARY_CHANNEL: 0,
            FALLBACK_CHANNEL: 0,
        }
        self._counters = {
            "attempts": 0,
            "successful_calls": 0,
            "retries": 0,
            "rate_limit_events": 0,
            "timeout_events": 0,
            "failures": 0,
            "fallbacks": 0,
            "fallback_successes": 0,
            "fallback_fused": 0,
            "active_requests": 0,
            "max_active_requests": 0,
        }
        self._routes = {
            PRIMARY_CHANNEL: _ChannelRoute(
                name=PRIMARY_CHANNEL,
                base_url=config.base_url,
                api_key=config.primary_api_key,
                model=config.model,
                max_concurrency=config.max_concurrency,
                min_concurrency=config.min_concurrency,
                global_max_concurrency=config.global_max_concurrency,
            ),
            FALLBACK_CHANNEL: _ChannelRoute(
                name=FALLBACK_CHANNEL,
                base_url=config.fallback_base_url,
                api_key=config.fallback_api_key,
                model=config.fallback_model,
                max_concurrency=config.fallback_max_concurrency,
                min_concurrency=config.fallback_min_concurrency,
                global_max_concurrency=config.fallback_global_max_concurrency,
            ),
        }
        self._global_slots = {
            PRIMARY_CHANNEL: _FileSlotLimiter(
                config.gate_dir,
                PRIMARY_CHANNEL,
                config.global_max_concurrency,
            ),
            FALLBACK_CHANNEL: _FileSlotLimiter(
                config.gate_dir,
                FALLBACK_CHANNEL,
                config.fallback_global_max_concurrency,
            ),
        }

    @property
    def fallback_enabled(self) -> bool:
        return self.config.fallback_enabled

    def current_channel(self) -> str:
        return _ACTIVE_CHANNEL.get()

    def route_for_channel(self, channel: str) -> _ChannelRoute:
        route = self._routes.get(channel)
        if route is None or not route.api_key:
            raise ValueError(f"DeepSeek {channel} route is unavailable")
        return route

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            result: dict[str, Any] = dict(self._counters)
            current_limit = self._limits[PRIMARY_CHANNEL]
            result.update(
                {
                    "concurrency_limit": current_limit,
                    "primary_concurrency_limit": self._limits[PRIMARY_CHANNEL],
                    "fallback_concurrency_limit": self._limits[FALLBACK_CHANNEL],
                    "min_concurrency": self.config.min_concurrency,
                    "fallback_min_concurrency": (self.config.fallback_min_concurrency),
                    "adaptive_concurrency": self.config.adaptive_concurrency,
                    "gate_scope": "process",
                    "worker_processes": self.config.worker_processes,
                    "process_primary_limit": self._limits[PRIMARY_CHANNEL],
                    "process_aggregate_hint": (
                        current_limit * self.config.worker_processes
                    ),
                    "global_concurrency_limit": (self.config.global_max_concurrency),
                    "fallback_global_concurrency_limit": (
                        self.config.fallback_global_max_concurrency
                    ),
                    "request_timeout_seconds": (
                        self.config.request_timeout_seconds
                    ),
                    "max_retries": self.config.max_retries,
                    "retry_base_seconds": self.config.retry_base_seconds,
                    "retry_jitter_ratio": self.config.retry_jitter_ratio,
                    "retry_max_seconds": self.config.retry_max_seconds,
                    "total_request_deadline_seconds": (
                        self.config.total_request_deadline_seconds
                    ),
                    "fallback_enabled": self.config.fallback_enabled,
                }
            )
            return result

    def _emit(self, event: str, **fields: Any) -> None:
        path = self.config.telemetry_path
        if path is None:
            return
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            "pid": os.getpid(),
            "gate_scope": "process",
            **fields,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        lock_path = path.with_name(path.name + ".lock")
        with lock_path.open("a", encoding="utf-8") as lock_stream:
            fcntl.flock(lock_stream, fcntl.LOCK_EX)
            try:
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(line)
            finally:
                fcntl.flock(lock_stream, fcntl.LOCK_UN)

    def _acquire_slot(self, channel: str) -> None:
        with self._condition:
            self._condition.wait_for(
                lambda: self._active_by_channel[channel] < self._limits[channel]
            )
            self._active_by_channel[channel] += 1
            self._counters["active_requests"] += 1
            self._counters["max_active_requests"] = max(
                self._counters["max_active_requests"],
                self._counters["active_requests"],
            )

    def _reduce_limit(self, channel: str, status: int | None) -> None:
        route = self._routes[channel]
        if (
            not self.config.adaptive_concurrency
            or self._limits[channel] <= route.min_concurrency
            or not (status == 429 or (status is not None and status >= 500))
        ):
            return
        now = time.monotonic()
        if now - self._last_reduction[channel] < self.config.adaptive_cooldown_seconds:
            return
        reduction = max(1, self._limits[channel] // 4)
        self._limits[channel] = max(
            route.min_concurrency,
            self._limits[channel] - reduction,
        )
        self._last_reduction[channel] = now
        self._success_streak[channel] = 0

    def _record_success(self, channel: str) -> None:
        route = self._routes[channel]
        if self._limits[channel] >= route.max_concurrency:
            self._success_streak[channel] = 0
            return
        self._success_streak[channel] += 1
        if self._success_streak[channel] >= self.config.successes_to_ramp:
            self._limits[channel] += 1
            self._success_streak[channel] = 0

    def _release_slot(
        self,
        channel: str,
        *,
        failed: bool = False,
        error_status: int | None = None,
    ) -> None:
        with self._condition:
            self._active_by_channel[channel] -= 1
            self._counters["active_requests"] -= 1
            if failed:
                self._reduce_limit(channel, error_status)
            else:
                self._record_success(channel)
            self._condition.notify_all()

    def _fuse_path(self) -> Path:
        return self.config.fallback_fuse_path or (
            self.config.gate_dir / "fallback.fused"
        )

    def fallback_fused(self) -> bool:
        path = self._fuse_path()
        try:
            age = time.time() - path.stat().st_mtime
        except FileNotFoundError:
            return False
        if age <= self.config.fallback_fuse_seconds:
            return True
        return False

    def _fuse_fallback(self, *, status: int | None, reason: str) -> None:
        path = self._fuse_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "reason": reason,
            "status_code": status,
        }
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
        with self._condition:
            self._counters["fallback_fused"] += 1

    def _simulator_pin(self) -> str:
        simulator = _CURRENT_SIMULATOR.get()
        if simulator is None:
            return PRIMARY_CHANNEL
        channel = getattr(simulator, "_deepseek_channel_pin", PRIMARY_CHANNEL)
        return (
            channel
            if channel in {PRIMARY_CHANNEL, FALLBACK_CHANNEL}
            else PRIMARY_CHANNEL
        )

    def _pin_simulator(self, channel: str) -> None:
        simulator = _CURRENT_SIMULATOR.get()
        if simulator is not None:
            try:
                setattr(simulator, "_deepseek_channel_pin", channel)
            except (AttributeError, TypeError):
                pass

    def _call_once(
        self,
        channel: str,
        function: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        attempt: int,
        fallback_count: int,
        deadline: float,
    ) -> Any:
        self.route_for_channel(channel)
        call_kwargs = dict(kwargs)
        call_kwargs["num_retries"] = 0
        self._acquire_slot(channel)
        descriptor = self._global_slots[channel].acquire()
        channel_token = _ACTIVE_CHANNEL.set(channel)
        started = time.perf_counter()
        error_status: int | None = None
        failed = False
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DeepSeekTotalDeadlineExceeded(
                    "DeepSeek request exceeded its total deadline"
                )
            call_kwargs["timeout"] = min(
                self.config.request_timeout_seconds,
                remaining,
            )
            with self._condition:
                self._counters["attempts"] += 1
            result = function(*args, **call_kwargs)
            if time.monotonic() > deadline:
                raise DeepSeekTotalDeadlineExceeded(
                    "DeepSeek request exceeded its total deadline"
                )
        except Exception as error:
            failed = True
            error_status = _status_code(error)
            timed_out = _is_timeout(error)
            with self._condition:
                if error_status == 429:
                    self._counters["rate_limit_events"] += 1
                if timed_out:
                    self._counters["timeout_events"] += 1
            if time.monotonic() > deadline and not isinstance(
                error,
                DeepSeekTotalDeadlineExceeded,
            ):
                error = DeepSeekTotalDeadlineExceeded(
                    "DeepSeek request exceeded its total deadline"
                )
                setattr(error, "_deepseek_attempt", attempt + 1)
                if not timed_out:
                    with self._condition:
                        self._counters["timeout_events"] += 1
            setattr(error, "_deepseek_channel", channel)
            setattr(
                error, "_deepseek_latency_ms", (time.perf_counter() - started) * 1000
            )
            raise error
        else:
            latency_ms = (time.perf_counter() - started) * 1000
            with self._condition:
                self._counters["successful_calls"] += 1
                if channel == FALLBACK_CHANNEL:
                    self._counters["fallback_successes"] += 1
            self._emit(
                "success",
                channel=channel,
                attempt=attempt,
                fallback_count=fallback_count,
                latency_ms=round(latency_ms, 2),
                concurrency_limit=self.snapshot()[f"{channel}_concurrency_limit"],
                global_concurrency_limit=(
                    self.config.fallback_global_max_concurrency
                    if channel == FALLBACK_CHANNEL
                    else self.config.global_max_concurrency
                ),
                worker_processes=self.config.worker_processes,
            )
            return result
        finally:
            _ACTIVE_CHANNEL.reset(channel_token)
            self._global_slots[channel].release(descriptor)
            self._release_slot(
                channel,
                failed=failed,
                error_status=error_status,
            )

    def _emit_retry(
        self,
        error: Exception,
        *,
        channel: str,
        attempt: int,
        delay: float,
        fallback_count: int,
    ) -> None:
        status = _status_code(error)
        self._emit(
            "retry",
            channel=channel,
            attempt=attempt,
            error_type=type(error).__name__,
            status_code=status,
            timeout=_is_timeout(error),
            fallback_count=fallback_count,
            fallback_events=fallback_count,
            latency_ms=round(
                float(getattr(error, "_deepseek_latency_ms", 0.0)),
                2,
            ),
            delay_seconds=round(delay, 3),
            concurrency_limit=self.snapshot()[f"{channel}_concurrency_limit"],
            global_concurrency_limit=(
                self.config.fallback_global_max_concurrency
                if channel == FALLBACK_CHANNEL
                else self.config.global_max_concurrency
            ),
            worker_processes=self.config.worker_processes,
        )

    def _fail(
        self,
        error: Exception,
        *,
        channel: str,
        attempt: int,
        fallback_count: int,
        fallback_fused: bool = False,
    ) -> None:
        with self._condition:
            self._counters["failures"] += 1
        self._emit(
            "failure",
            channel=channel,
            attempt=attempt,
            error_type=type(error).__name__,
            status_code=_status_code(error),
            timeout=_is_timeout(error),
            retryable=_is_retryable(error, _status_code(error)),
            fallback_count=fallback_count,
            fallback_events=fallback_count,
            fallback_fused=fallback_fused,
            latency_ms=round(
                float(getattr(error, "_deepseek_latency_ms", 0.0)),
                2,
            ),
            concurrency_limit=self.snapshot()[f"{channel}_concurrency_limit"],
        )

    def _try_primary(
        self,
        function: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        deadline: float,
    ) -> tuple[Any, None] | tuple[None, Exception]:
        retries = self.config.max_retries
        for attempt in range(retries + 1):
            try:
                result = self._call_once(
                    PRIMARY_CHANNEL,
                    function,
                    args,
                    kwargs,
                    attempt=attempt,
                    fallback_count=0,
                    deadline=deadline,
                )
            except Exception as error:
                setattr(error, "_deepseek_attempt", attempt + 1)
                status = _status_code(error)
                if isinstance(error, DeepSeekTotalDeadlineExceeded):
                    return None, error
                if _is_retryable(error, status) and attempt < retries:
                    with self._condition:
                        self._counters["retries"] += 1
                    delay = max(
                        _retry_delay(
                            self.config.retry_base_seconds,
                            self.config.retry_jitter_ratio,
                            attempt,
                            max_seconds=self.config.retry_max_seconds,
                        ),
                        _retry_after_seconds(
                            error,
                            max_seconds=self.config.retry_max_seconds,
                        ),
                    )
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None, DeepSeekTotalDeadlineExceeded(
                            "DeepSeek request exceeded its total deadline"
                        )
                    delay = min(delay, remaining)
                    self._emit_retry(
                        error,
                        channel=PRIMARY_CHANNEL,
                        attempt=attempt + 1,
                        delay=delay,
                        fallback_count=0,
                    )
                    if delay:
                        time.sleep(delay)
                    if time.monotonic() >= deadline:
                        return None, DeepSeekTotalDeadlineExceeded(
                            "DeepSeek request exceeded its total deadline"
                        )
                    continue
                return None, error
            else:
                self._pin_simulator(PRIMARY_CHANNEL)
                return result, None
        raise AssertionError("primary retry loop did not return")

    def _try_fallback(
        self,
        function: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        from_primary_error: Exception | None = None,
        deadline: float,
    ) -> Any:
        if not self.fallback_enabled:
            if from_primary_error is None:
                raise ValueError("DeepSeek fallback route is unavailable")
            raise from_primary_error
        if time.monotonic() >= deadline:
            error = DeepSeekTotalDeadlineExceeded(
                "DeepSeek request exceeded its total deadline"
            )
            self._fail(error, channel=FALLBACK_CHANNEL, attempt=0, fallback_count=1)
            raise error
        if self.fallback_fused():
            error = DeepSeekFallbackFusedError(
                "DeepSeek fallback is circuit-broken after an authentication "
                "or balance failure"
            )
            self._fail(error, channel=FALLBACK_CHANNEL, attempt=1, fallback_count=1)
            raise error
        if from_primary_error is not None:
            with self._condition:
                self._counters["fallbacks"] += 1
            self._emit(
                "fallback",
                channel=FALLBACK_CHANNEL,
                from_channel=PRIMARY_CHANNEL,
                attempt=1,
                error_type=type(from_primary_error).__name__,
                status_code=_status_code(from_primary_error),
                timeout=_is_timeout(from_primary_error),
                fallback_count=1,
                fallback_events=1,
            )
        try:
            result = self._call_once(
                FALLBACK_CHANNEL,
                function,
                args,
                kwargs,
                attempt=1,
                fallback_count=1,
                deadline=deadline,
            )
        except Exception as error:
            status = _status_code(error)
            fatal = _is_fallback_fatal(error, status)
            if fatal:
                reason = "balance" if status == 402 else "auth_or_balance"
                self._fuse_fallback(status=status, reason=reason)
            self._fail(
                error,
                channel=FALLBACK_CHANNEL,
                attempt=1,
                fallback_count=1,
                fallback_fused=fatal,
            )
            raise
        self._pin_simulator(FALLBACK_CHANNEL)
        return result

    def invoke(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        deadline = time.monotonic() + self.config.total_request_deadline_seconds
        preferred = self._simulator_pin()
        if preferred == FALLBACK_CHANNEL:
            return self._try_fallback(
                function,
                args,
                kwargs,
                deadline=deadline,
            )

        result, primary_error = self._try_primary(
            function,
            args,
            kwargs,
            deadline=deadline,
        )
        if primary_error is None:
            return result
        status = _status_code(primary_error)
        if (
            self.fallback_enabled
            and time.monotonic() < deadline
            and _is_retryable(primary_error, status)
        ):
            return self._try_fallback(
                function,
                args,
                kwargs,
                from_primary_error=primary_error,
                deadline=deadline,
            )
        self._fail(
            primary_error,
            channel=PRIMARY_CHANNEL,
            attempt=int(getattr(primary_error, "_deepseek_attempt", 1)),
            fallback_count=0,
        )
        raise primary_error


_GATE_LOCK = threading.Lock()
_GATE: DeepSeekRequestGate | None = None
_ORIGINAL_COMPLETION: Callable[..., Any] | None = None
_ORIGINAL_USER_NEXT: Callable[..., Any] | None = None


def configure_deepseek_client(
    config: DeepSeekClientConfig | None = None,
) -> DeepSeekRequestGate:
    global _GATE, _ORIGINAL_COMPLETION, _ORIGINAL_USER_NEXT
    resolved = config or DeepSeekClientConfig.from_env()
    with _GATE_LOCK:
        if _GATE is not None:
            if _GATE.config != resolved:
                raise RuntimeError(
                    "DeepSeek client was already configured with different settings"
                )
            return _GATE

        gate = DeepSeekRequestGate(resolved)
        from tau2.user import user_simulator
        from tau2.user.user_simulator import UserSimulator
        from tau2.utils import llm_utils

        original_generate = user_simulator.generate

        def gated_generate(*args: Any, **kwargs: Any) -> Any:
            return gate.invoke(original_generate, *args, **kwargs)

        user_simulator.generate = gated_generate

        if _ORIGINAL_COMPLETION is None:
            _ORIGINAL_COMPLETION = llm_utils.completion

            def routed_completion(*args: Any, **kwargs: Any) -> Any:
                channel = _ACTIVE_CHANNEL.get()
                model = kwargs.get("model")
                if isinstance(model, str) and model.startswith("deepseek/"):
                    route = gate.route_for_channel(channel)
                    kwargs = dict(kwargs)
                    kwargs["api_base"] = route.base_url
                    kwargs["api_key"] = route.api_key
                    kwargs["model"] = f"deepseek/{route.model}"
                return _ORIGINAL_COMPLETION(*args, **kwargs)

            llm_utils.completion = routed_completion

        if _ORIGINAL_USER_NEXT is None:
            _ORIGINAL_USER_NEXT = UserSimulator._generate_next_message

            def pinned_generate_next_message(
                simulator: Any,
                message: Any,
                state: Any,
            ) -> Any:
                token = _CURRENT_SIMULATOR.set(simulator)
                try:
                    return _ORIGINAL_USER_NEXT(simulator, message, state)
                finally:
                    _CURRENT_SIMULATOR.reset(token)

            UserSimulator._generate_next_message = pinned_generate_next_message

        _GATE = gate
        return gate
