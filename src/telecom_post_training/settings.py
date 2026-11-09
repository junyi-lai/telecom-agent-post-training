from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]


def _allow_insecure_http_from_env() -> bool:
    raw = os.getenv("ALLOW_INSECURE_DEEPSEEK_HTTP", "0").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        "ALLOW_INSECURE_DEEPSEEK_HTTP must be one of 0, 1, false, true, no, yes"
    )


def validate_deepseek_base_url(
    value: str,
    *,
    allow_insecure_http: bool | None = None,
) -> str:
    parsed = urlparse(value)
    if parsed.scheme == "https" and parsed.hostname:
        return value
    if parsed.scheme == "http" and parsed.hostname:
        if parsed.hostname == "localhost":
            return value
        try:
            address = ipaddress.ip_address(parsed.hostname)
        except ValueError:
            pass
        else:
            if address.is_loopback or address.is_private:
                return value
        if allow_insecure_http is None:
            allow_insecure_http = _allow_insecure_http_from_env()
        if allow_insecure_http:
            return value
    raise ValueError(
        "DEEPSEEK_BASE_URL must use HTTPS. Plain HTTP is allowed only for "
        "loopback or private-VPC addresses, through a trusted SSH tunnel, or "
        "with the explicit ALLOW_INSECURE_DEEPSEEK_HTTP=1 override."
    )


@dataclass(frozen=True)
class Settings:
    student_model: str
    student_model_revision: str
    student_model_path: Path
    vllm_model_alias: str
    deepseek_base_url: str
    deepseek_api_key: str
    deepseek_model: str
    deepseek_teacher_model: str
    tau2_repo: Path
    tau2_domain: str = "telecom"
    tau2_rollout_max_turns: int = 16
    tau2_rollout_context_tokens: int = 8192
    tau2_rollout_concurrency: int = 2

    @classmethod
    def from_env(cls) -> "Settings":
        tau2_repo = Path(
            os.getenv("TAU2_REPO", str(ROOT / "third_party" / "tau2-bench"))
        ).expanduser()
        if not tau2_repo.is_absolute():
            tau2_repo = (ROOT / tau2_repo).resolve()
        return cls(
            student_model=os.getenv("STUDENT_MODEL", "Qwen/Qwen3.5-2B"),
            student_model_revision=os.getenv("STUDENT_MODEL_REVISION", ""),
            student_model_path=(
                ROOT / os.getenv("STUDENT_MODEL_PATH", "models/Qwen3.5-2B")
            ).resolve(),
            vllm_model_alias=os.getenv("VLLM_MODEL_ALIAS", "student"),
            deepseek_base_url=os.getenv(
                "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"
            ),
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
            deepseek_teacher_model=os.getenv(
                "DEEPSEEK_TEACHER_MODEL",
                os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
            ),
            tau2_repo=tau2_repo,
            tau2_domain=os.getenv("TAU2_DOMAIN", "telecom"),
            tau2_rollout_max_turns=int(os.getenv("TAU2_ROLLOUT_MAX_TURNS", "16")),
            tau2_rollout_context_tokens=int(
                os.getenv("TAU2_ROLLOUT_CONTEXT_TOKENS", "8192")
            ),
            tau2_rollout_concurrency=int(os.getenv("TAU2_ROLLOUT_CONCURRENCY", "2")),
        )
