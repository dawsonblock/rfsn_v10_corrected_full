"""Entry point for ``python -m rfsn_v10``."""

from __future__ import annotations

import sys

from rfsn_v10.health import get_health_checker


def main() -> int:
    """Print a basic status summary and exit."""
    checker = get_health_checker()
    report = checker.get_health_report()
    status = report.get("overall_status", "unknown")
    print(f"RFSN v10 status: {status}")
    for name, item in report.items():
        if isinstance(item, dict):
            s = item.get("status", "unknown")
            print(f"  {name}: {s}")
    return 0 if status == "healthy" else 1


if __name__ == "__main__":
    sys.exit(main())
