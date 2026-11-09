from __future__ import annotations

from telecom_post_training.deepseek_client import configure_deepseek_client


def main() -> int:
    configure_deepseek_client()
    from tau2.cli import main as tau2_main

    tau2_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
