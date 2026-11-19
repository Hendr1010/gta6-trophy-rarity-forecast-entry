[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Hendr1010/gta6-trophy-rarity-forecast-entry/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Hendr1010/gta6-trophy-rarity-forecast-entry/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecaster

Forecasts, for each of the 55 PlayStation 5 trophies of Grand Theft Auto VI,
five quantiles of its earn rate on Exophase one week after launch
(2026-11-26T00:00:00Z), as a percent.

## How it works

A trophy's rate is the fraction of the players Exophase tracks who have
earned it. That fraction follows a saturating curve in launch age: players
progress through the game, and new players joining the tracked population
dilute every rate at the same time.

The program (`main.py`, Python 3.13, standard library only) reads one case
file at a time and writes `<id>.out`:

1. **Log-time trend (primary model).** For each trophy, `logit(rate)` is
   fit by least squares against `u = ln(1 + hours since launch)` over the
   last 24 hours of distinct reads (Exophase only changes its page at its
   own refreshes, so identical consecutive reads are collapsed first), and
   the line is read at the target. A rate saturates in launch age, and a
   straight line in log-time is the shape that captures it without
   extrapolating launch-night growth past the horizon.
2. **Spread.** A random walk widening with the horizon (`0.5 * sqrt(days)`)
   combined with the trend fit's residual, so the 5%-95% band covers the
   refresh-and-dilution risk the history shows.
3. **Flat history.** When the history never saw the page refresh (the
   launch-night practice cases), the predictive distribution is bimodal:
   either no refresh lands in the horizon or one does and the tracked
   population jumps several-fold. The median hedges halfway between the
   two modes and the lower tail is widened to cover the refresh case.
4. **Saturated trophy.** A trophy the last read shows at 100% is forecast
   just below 100% with a narrow band: every tracked player holds it.

The program is deterministic (no randomness, no network) and runs in
~0.2 s per case, well inside the 8 s limit.

## Usage

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Input: `<id>.in` JSON with `trophies` (id, name, description, tier,
points, secret) and `history` (reads of `players_tracked` and `rates`).
Output: `<id>.out` JSON `{"trophies": [{"id": n, "q": [q05, q25, q50, q75,
q95]}, ...]}`, five finite numbers in [0, 100], non-decreasing.

## Validation

On three synthetic final-like cases (a launch-night ODE world, a
logistic-in-log-time world, and a realistic saturation world with the
tracked population growing 638 -> ~1.5M), the model scores a mean pinball
loss of 7.59 / 0.25 / 0.24, essentially tied with the house curve baseline
(7.58 / 0.24 / 0.23) and well below the persistence baseline
(8.27 / 0.28 / 0.25). On the flat practice histories it hedges for the
refresh dilution the persistence baseline misses entirely (house baseline
practice loss ~1.51).

## License

MIT. See [LICENSE](LICENSE).
