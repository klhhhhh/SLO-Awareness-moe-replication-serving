# src/moe_slo/harness/metrics.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from moe_slo.adapters.sglang_client import RequestRecord


def percentile(x: np.ndarray, p: float) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, p))


@dataclass
class Summary:
    n: int
    slo_s: float
    viol_rate: float

    ttft_p50: float
    ttft_p90: float
    ttft_p99: float

    e2e_p50: float
    e2e_p90: float
    e2e_p99: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def summarize(records: List[RequestRecord], slo_s: float) -> Summary:
    ttft = np.array([r.ttft_s for r in records], dtype=np.float64)
    e2e = np.array([r.e2e_s for r in records], dtype=np.float64)
    viol = float(np.mean(e2e > slo_s)) if e2e.size > 0 else float("nan")

    return Summary(
        n=len(records),
        slo_s=float(slo_s),
        viol_rate=viol,

        ttft_p50=percentile(ttft, 50),
        ttft_p90=percentile(ttft, 90),
        ttft_p99=percentile(ttft, 99),

        e2e_p50=percentile(e2e, 50),
        e2e_p90=percentile(e2e, 90),
        e2e_p99=percentile(e2e, 99),
    )


@dataclass
class WindowPoint:
    t_end: float
    n: int
    e2e_p99: float
    viol_rate: float


def window_series(
    records: List[RequestRecord],
    slo_s: float,
    window_s: float = 1.0,
) -> List[WindowPoint]:
    """
    Simple time-series: group by last_ts into fixed windows.
    For burst plots, this is usually "good enough".
    """
    if not records:
        return []

    t0 = min(r.send_ts for r in records)
    t_max = max(r.last_ts for r in records)
    num_win = int(np.ceil((t_max - t0) / window_s))

    buckets: List[List[RequestRecord]] = [[] for _ in range(num_win)]
    for r in records:
        idx = int((r.last_ts - t0) // window_s)
        idx = min(max(idx, 0), num_win - 1)
        buckets[idx].append(r)

    points: List[WindowPoint] = []
    for i, rs in enumerate(buckets):
        if not rs:
            points.append(WindowPoint(t_end=(i + 1) * window_s, n=0, e2e_p99=float("nan"), viol_rate=float("nan")))
            continue
        e2e = np.array([x.e2e_s for x in rs], dtype=np.float64)
        viol = float(np.mean(e2e > slo_s))
        points.append(
            WindowPoint(
                t_end=(i + 1) * window_s,
                n=len(rs),
                e2e_p99=percentile(e2e, 99),
                viol_rate=viol,
            )
        )
    return points
