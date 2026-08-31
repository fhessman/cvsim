# utils.py
"""Small cross-module helpers with no dependencies on the rest of the package."""

import time


def with_progress(iterable, n, label, step=10):
    """
    Yield items from `iterable` (of known length n), printing a
    "label: NN% (elapsed Ns)" line to stdout at each `step`-percent
    milestone (default every 10%) -- so a long, non-interactive loop (the
    disc irradiation integral, a light-curve phase sweep, "direct" pixel
    mapping's per-sample splat) doesn't look hung. Elapsed time is
    wall-clock since the first item was requested (time.monotonic, so
    it's unaffected by system clock adjustments).
    """
    start = time.monotonic()
    next_pct = step
    for i, item in enumerate(iterable):
        yield item
        if n > 0:
            pct = 100.0 * (i + 1) / n
            if pct >= next_pct:
                elapsed = time.monotonic() - start
                print(f"  {label}: {int(pct)}% (elapsed {elapsed:.1f}s)", flush=True)
                next_pct += step
