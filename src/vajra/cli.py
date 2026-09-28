"""Command line: ``vajra evaluate`` scores the baselines; ``vajra cap-example`` prints an alert."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

import numpy as np

from vajra.data.synthetic import generate
from vajra.grid import Grid, within
from vajra.nowcast import lightning_only, persistence, radar_extrapolation
from vajra.verify import (
    Contingency,
    brier_score,
    contingency,
    first_event_step,
    fractions_skill_score,
    initiation_mask,
    minutes_gained,
    warning_lead_minutes,
)
from vajra.warn import build_cap, draft_polygons, risk_levels

IST = timezone(timedelta(hours=5, minutes=30))
METHODS = ("Persistence", "Lightning-only tracking", "Radar extrapolation")


def evaluate_synthetic(seed: int = 0, leads=(30, 60, 120), radius_km: float = 10.0, threshold: float = 0.5):
    """Score the three baselines on synthetic storms.

    Returns (rows, minutes_gained, new_storm_pixels), where each row is
    (method, lead_minutes, critical_success_index, fractions_skill_score, brier_score).
    """
    storms = generate(seed=seed, minutes=300, step=10)
    km, step = storms.grid.km, storms.step_minutes
    r_px = radius_km / km
    observed = within(storms.flashes > 0, radius_km, km)
    n_t = len(storms.minutes)
    n_max = int(max(leads) // step)
    horizon = int(60 // step)
    window = int(2 * round(r_px) + 1)
    acc = {(m, lead): (Contingency(0, 0, 0, 0), [], []) for m in METHODS for lead in leads}
    warned = {m: np.zeros_like(observed) for m in METHODS}
    for t in range(1, n_t - 1):
        forecasts = {
            "Persistence": persistence(observed[t], n_max),
            "Lightning-only tracking": lightning_only(storms.flashes[t - 1 : t + 1], n_max, r_px),
            "Radar extrapolation": radar_extrapolation(storms.reflectivity[t - 1 : t + 1], n_max, r_px),
        }
        for m in METHODS:
            warned[m][t] = (forecasts[m][:horizon] >= threshold).any(axis=0)
            for lead in leads:
                k = int(lead // step)
                if t + k >= n_t:
                    continue
                cont, fs, os_ = acc[(m, lead)]
                p = np.clip(forecasts[m][k - 1], 0, 1)
                hit = contingency(p >= threshold, observed[t + k])
                acc[(m, lead)] = (cont + hit, [*fs, p], [*os_, observed[t + k]])
    rows = []
    for m in METHODS:
        for lead in leads:
            cont, fs, os_ = acc[(m, lead)]
            f, o = np.stack(fs), np.stack(os_)
            rows.append((m, lead, cont.csi, fractions_skill_score(f, o, window), brier_score(f, o)))
    first = first_event_step(observed)
    new = initiation_mask(observed, first, radius_px=3 * r_px, lookback=3)
    lead_min = {m: warning_lead_minutes(warned[m], first, step, horizon) for m in METHODS}
    gained = minutes_gained(lead_min["Radar extrapolation"], lead_min["Lightning-only tracking"], new)
    return rows, gained, int(new.sum())


def cap_example() -> str:
    """An orange warning over a made-up area near Chunar, marked as an exercise."""
    grid = Grid(nx=64, ny=64, km=2.0, lon0=82.55, lat0=24.8)
    yy, xx = np.mgrid[0:64, 0:64]
    chance = np.exp(-((xx - 32) ** 2 + (yy - 30) ** 2) / (2 * 12.0**2))
    levels = risk_levels(chance, np.full(chance.shape, 0.8))
    polygons = [p["polygon"] for p in draft_polygons(levels, grid) if p["level"] == 2]
    return build_cap(2, polygons, "Mirzapur district", datetime(2026, 5, 13, 13, 42, tzinfo=IST))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vajra", description="Lightning and thunderstorm nowcasting for India."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("evaluate", help="score the baselines on synthetic storms")
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--leads", type=int, nargs="+", default=[30, 60, 120], help="lead times in minutes")
    sub.add_parser("cap-example", help="print an example CAP alert, marked as an exercise")
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        rows, gained, n_new = evaluate_synthetic(args.seed, tuple(args.leads))
        print("Synthetic storms: these numbers check the code, not forecast skill on real weather.\n")
        print(f"{'Method':<26}{'Lead':>6}{'CSI':>7}{'FSS':>7}{'Brier':>8}")
        for m, lead, csi, fss, bs in rows:
            print(f"{m:<26}{lead:>4} m{csi:>7.2f}{fss:>7.2f}{bs:>8.3f}")
        print(
            f"\nFor {n_new} pixels hit by new storms, radar extrapolation warned {gained:.1f} minutes "
            "earlier than lightning-only tracking, on average."
        )
    else:
        print(cap_example())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
