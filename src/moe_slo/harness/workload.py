# src/moe_slo/harness/workload.py
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Literal, Optional


@dataclass
class PoissonWorkload:
    qps: float
    duration_s: float
    seed: int = 0

    def schedule(self) -> List[float]:
        """
        Returns a list of send time offsets (seconds) from t=0.
        """
        rng = random.Random(self.seed)
        t = 0.0
        times: List[float] = []
        while t < self.duration_s:
            if self.qps <= 0:
                break
            gap = rng.expovariate(self.qps)
            t += gap
            if t <= self.duration_s:
                times.append(t)
        return times


@dataclass
class BurstWorkload:
    qps_on: float
    qps_off: float
    on_s: float
    off_s: float
    cycles: int
    seed: int = 0

    def schedule(self) -> List[float]:
        """
        Alternates ON and OFF phases. Returns send time offsets from t=0.
        """
        rng = random.Random(self.seed)
        times: List[float] = []
        t0 = 0.0

        for _ in range(self.cycles):
            # ON phase
            t = 0.0
            while t < self.on_s:
                if self.qps_on > 0:
                    t += rng.expovariate(self.qps_on)
                    if t < self.on_s:
                        times.append(t0 + t)
                else:
                    break
            t0 += self.on_s

            # OFF phase
            t = 0.0
            while t < self.off_s:
                if self.qps_off > 0:
                    t += rng.expovariate(self.qps_off)
                    if t < self.off_s:
                        times.append(t0 + t)
                else:
                    break
            t0 += self.off_s

        times.sort()
        return times

    @property
    def duration_s(self) -> float:
        return self.cycles * (self.on_s + self.off_s)
