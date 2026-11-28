"""HTTP health probe for container HEALTHCHECKs (standard library only).

Slim Python images ship neither curl nor wget, so the Docker healthchecks run this script::

    python scripts/healthcheck.py http://127.0.0.1:8000/health
    python scripts/healthcheck.py http://127.0.0.1:8050/ --timeout 3

Exit codes: 0 the URL answered with an HTTP status below 400, 1 anything else (connection refused,
timeout, 4xx/5xx) - the reason goes to stderr, which ``docker inspect`` shows in the health log.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from collections.abc import Sequence

EXIT_HEALTHY = 0
EXIT_UNHEALTHY = 1
DEFAULT_TIMEOUT_S = 3.0


def check(url: str, timeout: float = DEFAULT_TIMEOUT_S) -> tuple[bool, str]:
    """Probe ``url``; return ``(healthy, detail)``."""
    if not url.lower().startswith(("http://", "https://")):
        return False, f"not an http(s) URL: {url}"
    try:
        # The scheme is checked above, so this cannot be turned into a file:// read.
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            status = int(response.status)
    except urllib.error.HTTPError as error:
        return False, f"HTTP {error.code} from {url}"
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return False, f"{url} unreachable: {error}"
    return (status < 400), f"HTTP {status} from {url}"


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description="Container health probe.")
    parser.add_argument("url", help="URL to request, e.g. http://127.0.0.1:8000/health")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="seconds")
    args = parser.parse_args(argv)
    healthy, detail = check(args.url, args.timeout)
    if not healthy:
        print(detail, file=sys.stderr)
    return EXIT_HEALTHY if healthy else EXIT_UNHEALTHY


if __name__ == "__main__":
    sys.exit(main())
