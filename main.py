"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Forecasts, for each of the game's PS5 trophies, five quantiles of its earn
rate on Exophase at the case's target, as a percent.

Model (earned / tracked decomposition)
--------------------------------------
A trophy's rate is a fraction: rate = 100 * earned / tracked.  The two
counts move for different reasons, so they are forecast separately and
divided at the target:

* tracked players P(t): the Exophase-tracked population.  At launch it
  grows fast and decelerates.  Its log growth rate gP is fit from the
  case's history (least squares on log P against time over the distinct
  reads); when the history never saw the population move (the page had
  not refreshed yet), a launch-week prior is used instead.
* earned counts E_i(t): how many tracked players hold trophy i.  Its log
  growth rate gE_i is fit per trophy from the history and shrunk toward
  the cross-trophy mean, because single-trophy counts are small and
  noisy.  When a trophy never moved in the history, a tier prior is used
  (story/early trophies are earned faster than rare ones).

Both growth rates are damped over the horizon: growth over H days is
modelled with an effective horizon H_eff = TAU * (1 - exp(-H / TAU)),
so a 4.5-day forecast does not extrapolate launch-night growth linearly.

The spread is honest about a launch-week forecast: a floor that grows
with the horizon, plus a term from the fit residuals, so the 5%-95% band
covers the refresh-and-dilution risk the history shows (Exophase
refreshes on its own schedule; a refresh grows the tracked population
and pulls every rate down unless the earned count kept pace).

Quantiles are placed on the logit scale: q_tau = logistic(logit(R) +
z_tau * sigma), rounded to 4 decimals half-up.  Deterministic: no
randomness, no network, standard library only.

Input: every <id>.in in the input directory, JSON
{"schema": "makefun/gta6-trophies-case/v1", "id", "at",
"horizon_minutes", "target", "levels", "trophies", "history"}.
`history` is every read before `at`, oldest first, each
{slot, at, players_tracked, trophies_earned, rates} with `rates` in
trophy order (null where the read lacked the trophy).

Output: <id>.out per case, {"trophies": [{"id": n, "q": [q05, q25,
q50, q75, q95]}, ...]}, one row per trophy, five finite numbers in
[0, 100], non-decreasing.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- Tunable constants (fixed in advance, never tuned on the answers) ---

# Tracked-population log growth prior (per day) when the history never
# saw the population move. Launch week: the tracked population grows
# roughly an order of magnitude in the first days.
G_P_PRIOR = 1.2
# Earned-count log growth prior (per day) by tier when a trophy never
# moved in the history. Early/story trophies are earned fastest.
ALPHA_PRIOR = {"platinum": 0.05, "gold": 0.4, "silver": 0.8, "bronze": 1.2}
# Damping timescale (days) for growth extrapolation over the horizon,
# used only when the history never moved (the prior path).
TAU_G = 1.5
# Launch of the game (UTC).  Growth is fit against log-time u =
# ln(1 + hours since launch): player counts and earned counts grow
# roughly log-linearly in launch age, so a straight line in u already
# decelerates -- a line in plain time extrapolates launch-night growth
# far past the horizon and is the single biggest error source.
LAUNCH_ISO = "2026-11-19T00:00:00Z"
# Fit window for the growth rates: the most recent 24 hours of distinct
# states (the current regime, not launch night) -- the same window the
# house curve baseline uses.  Fewer than 3 states in the window falls
# back to every state.
FIT_WINDOW_HOURS = 24.0
# Shrinkage weight on a trophy's own fitted earned-growth rate (the
# rest goes to the cross-trophy mean).
OWN_WEIGHT = 0.6
# Spread (logit units): sigma = SIGMA_FLOOR + SIGMA_SLOPE * sqrt(H + SIGMA_H)
SIGMA_FLOOR = 0.35
SIGMA_SLOPE = 0.45
SIGMA_H = 0.5
# Extra spread per unit of residual misfit of the earned-count fit.
SIGMA_RESID = 0.5
# A trophy the last read shows at (essentially) 100%: its median is set
# just below 100 with a narrow band (every tracked player holds it, and
# the launch-night refresh left it at 100%).
SATURATED_MEDIAN = 99.9
SATURATED_SIGMA = 0.4
# When the history never saw the page refresh, the predictive distribution
# is bimodal: either no refresh lands in the horizon (rate near the last
# read) or one does (the tracked population jumps several-fold and the
# rate drops hard).  The median hedges halfway between the two modes, and
# the lower tail is widened to cover the refresh case while the upper tail
# stays narrow (rates rarely jump up sharply at a refresh).
FLAT_MEDIAN_HEDGE = 0.5
FLAT_SIGMA_LOW_MULT = 2.2


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def logistic(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def round4(x):
    """Round to 4 decimals, half up, as JavaScript's Math.round does."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s):
    """ISO-8601 timestamp (with Z) -> epoch seconds (float)."""
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def linfit(xs, ys):
    """Least-squares slope and intercept of y on x; returns (slope, intercept).

    Returns (0.0, mean(y)) when the fit is degenerate.
    """
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return 0.0, my
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return slope, my - slope * mx


def distinct_states(history):
    """Collapse consecutive reads with identical (players, rates) into states.

    Exophase only changes its page at its own refreshes, so long runs of
    reads are identical.  Returns a list of (epoch_seconds, players,
    rates).
    """
    states = []
    for read in history:
        t = parse_time(read["at"])
        players = read["players_tracked"]
        rates = read["rates"]
        if states:
            _, p_prev, r_prev = states[-1]
            if p_prev == players and r_prev == rates:
                continue
        states.append((t, players, rates))
    return states


def forecast_case(case):
    trophies = case["trophies"]
    history = case["history"]
    horizon_days = case["horizon_minutes"] / 1440.0
    n = len(trophies)

    if not history:
        raise ValueError("case has no history")

    states = distinct_states(history)
    last_rates = states[-1][2]
    p_last = float(states[-1][1])
    t_last = states[-1][0]
    t_target = parse_time(case["target"])
    launch = parse_time(LAUNCH_ISO)

    # Log-time u = ln(1 + hours since launch).  Growth is fit against u,
    # so the extrapolation to the target is already damped: a straight
    # line in plain time extrapolates launch-night growth far past the
    # horizon and is the single biggest error source.
    def u_of(t):
        return math.log1p(max(0.0, (t - launch) / 3600.0))

    u_last = u_of(t_last)
    u_target = u_of(t_target)
    du = max(0.0, u_target - u_last)

    # Fit window: the most recent FIT_WINDOW_HOURS of distinct states.
    win_start = t_last - FIT_WINDOW_HOURS * 3600.0
    win = [s for s in states if s[0] >= win_start]
    if len(win) < 3:
        win = states
    win_u = [u_of(s[0]) for s in win]
    win_p = [float(s[1]) for s in win]
    flat = len(states) < 2

    # Effective horizon with damped growth (prior path only).
    h_eff = TAU_G * (1.0 - math.exp(-horizon_days / TAU_G)) if horizon_days > 0 else 0.0

    # --- Model A: log-time trend of the rate itself. ---
    # logit(rate) is fit against u per trophy; the regression line is
    # read at the target.  No shrinkage: trophies genuinely differ in
    # speed (a platinum crawls, an early story trophy sprints), and
    # shrinking their slopes toward the mean biases both.  The line is
    # used as fit (not re-anchored at the last read): the last read's
    # 2-decimal rate carries quantization noise, and the regression
    # averages it out.  This is the strongest single model for a
    # saturating rate.
    trend_logit = []
    trend_resid = []
    for i in range(n):
        series = [(win_u[k], win[k][2][i]) for k in range(len(win))
                  if win[k][2][i] is not None]
        if len(series) >= 2 and series[-1][1] != series[0][1]:
            xs = [t for t, _ in series]
            ys = [logit(r) for _, r in series]
            slope, intercept = linfit(xs, ys)
            slope = min(3.0, max(-3.0, slope))
            pred = [intercept + slope * x for x in xs]
            r2 = sum((y - p) ** 2 for y, p in zip(ys, pred)) / max(1, len(ys))
            trend_resid.append(math.sqrt(r2))
            trend_logit.append(intercept + slope * u_target)
        else:
            trend_resid.append(0.0)
            trend_logit.append(None)

    # --- Model B: earned / tracked decomposition in log-time. ---
    # log P and log E_i are fit against u; the rate at the target is
    # their ratio.  This sees dilution explicitly: when the tracked
    # population grows faster than the earned counts, the rate falls.
    if len(win) >= 2 and win_p[-1] != win_p[0]:
        log_p = [math.log(max(1.0, p)) for p in win_p]
        g_p, _ = linfit(win_u, log_p)
        g_p = min(4.0, max(0.0, g_p))
        p_target = p_last * math.exp(g_p * du)
        counts_ok = True
    else:
        g_p = None
        p_target = p_last * math.exp(G_P_PRIOR * h_eff)
        counts_ok = False

    earned = []
    for s in win:
        p = float(s[1])
        earned.append([(r * p / 100.0) if r is not None else None for r in s[2]])

    g_e = []
    counts_resid = []
    for i in range(n):
        series = [(win_u[k], earned[k][i]) for k in range(len(win))
                  if earned[k][i] is not None]
        if len(series) >= 2 and series[-1][1] != series[0][1]:
            xs = [t for t, _ in series]
            ys = [math.log(max(0.5, e)) for _, e in series]
            slope, intercept = linfit(xs, ys)
            pred = [intercept + slope * x for x in xs]
            r2 = sum((y - p) ** 2 for y, p in zip(ys, pred)) / max(1, len(ys))
            counts_resid.append(math.sqrt(r2))
            g_e.append(min(5.0, max(0.0, slope)))
        else:
            counts_resid.append(0.0)
            g_e.append(ALPHA_PRIOR.get(trophies[i].get("tier", "bronze"), 1.0))

    g_mean = sum(g_e) / max(1, len(g_e))
    g_e = [OWN_WEIGHT * g + (1.0 - OWN_WEIGHT) * g_mean for g in g_e]

    # --- Spread (logit units). ---
    sigma_base = SIGMA_FLOOR + SIGMA_SLOPE * math.sqrt(horizon_days + SIGMA_H)

    rows = []
    for i, trophy in enumerate(trophies):
        r_last = last_rates[i]
        if r_last is None:
            r_last = 0.01
        r_last = float(r_last)

        if r_last >= 99.99:
            # Saturated trophy: everyone tracked holds it.
            median = SATURATED_MEDIAN
            sigma_low = SATURATED_SIGMA
            sigma_high = SATURATED_SIGMA
        elif flat or not counts_ok or trend_logit[i] is None:
            # No usable trend in the history: prior path.
            e_last = max(0.5, r_last * p_last / 100.0)
            e_target = e_last * math.exp(g_e[i] * h_eff)
            median = clamp_rate(100.0 * e_target / max(1.0, p_target))
            if flat:
                # Hedge between the no-refresh and refresh modes.
                median = clamp_rate(median * FLAT_MEDIAN_HEDGE)
            sigma_low = sigma_base + SIGMA_RESID * counts_resid[i]
            sigma_high = sigma_low
            if flat:
                sigma_low = sigma_low * FLAT_SIGMA_LOW_MULT
            sigma_low = min(2.5, sigma_low)
            sigma_high = min(2.0, sigma_high)
        else:
            # The log-time rate trend is the primary model: a trophy's rate
            # follows a saturating curve in launch age, and logit(rate)
            # against log-time is the straight line that captures it.
            median_logit = trend_logit[i]
            median = clamp_rate(100.0 * logistic(median_logit))
            # Spread: a random walk widening with the horizon, combined
            # with the trend fit's residual.  No wider: the counts split
            # is not used here (splitting two saturating logistics and
            # dividing them adds error), and an honest spread is the one
            # the fit itself supports.
            sigma = math.sqrt((0.5 * math.sqrt(horizon_days)) ** 2
                              + trend_resid[i] ** 2)
            sigma = max(0.3, sigma)
            sigma_low = min(2.0, sigma)
            sigma_high = sigma_low

        lg = logit(median)
        qs = []
        for k, z in enumerate(Z):
            sigma = sigma_low if z < 0 else sigma_high
            q = 100.0 * logistic(lg + z * sigma)
            qs.append(round4(min(100.0, max(0.0, q))))
        # Keep the row non-decreasing after rounding.
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})

    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = OUTPUT_DIR / (path.stem + ".out")
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
