"""Helper utilities for assigning per-worker environment variables."""
from __future__ import annotations

import os
import random
import socket
from typing import Dict

# Track reserved ports in the current launcher process so that we do not hand
# out the same port to multiple workers when they are spawned in quick
# succession.
_RESERVED_PORTS = set()


def _is_port_free(port: int) -> bool:
    """Return ``True`` if ``port`` appears to be available on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _reserve_port(base: int, step: int) -> int:
    """Find a free port starting at ``base`` with increments of ``step``."""
    if step <= 0:
        step = 1

    candidate = base
    for _ in range(1024):
        if candidate not in _RESERVED_PORTS and _is_port_free(candidate):
            _RESERVED_PORTS.add(candidate)
            return candidate
        candidate += step

    raise RuntimeError(
        f"Unable to locate a free port after scanning 1024 candidates starting at {base}"
    )


def prepare_worker_env(worker_idx: int, total_workers: int) -> Dict[str, str]:
    """Return environment overrides for ``worker_idx``.

    Parameters
    ----------
    worker_idx:
        Zero-based worker index assigned by the launcher.
    total_workers:
        Total number of workers that will be created.
    """

    base = int(os.environ.get("VLLM_TORCH_DIST_INIT_PORT_BASE", "29500"))
    step = int(os.environ.get("VLLM_TORCH_DIST_INIT_PORT_STEP", "10"))
    jitter = int(os.environ.get("VLLM_TORCH_DIST_INIT_PORT_JITTER", "0"))

    if jitter > 0:
        base += random.randint(0, jitter)

    port = _reserve_port(base + worker_idx * step, step)

    env = {
        "MASTER_ADDR": os.environ.get("MASTER_ADDR", "127.0.0.1"),
        "MASTER_PORT": str(port),
        "VLLM_TORCH_DIST_INIT_PORT": str(port),
        "VLLM_WORKER_ASSIGNED_PORT": str(port),
        "VLLM_WORKER_INDEX": str(worker_idx),
        "VLLM_WORKER_TOTAL": str(total_workers),
    }

    return env
