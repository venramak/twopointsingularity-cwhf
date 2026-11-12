"""
Sensitivity and operating-characteristic checks requested in review (Signal != Value, v1.1).

  A. Power of the H1 primary test by between-person spread of D_i (analytic, noncentral t).
  B. The blinded sample-size rule: false-alarm rate and power of the full adaptive procedure.
  C. Side bias: pressure that pushes choices toward one screen side, with the attractor's side
     randomised per item per person, must not produce a positive D.
  D. H2: how the source-fix result depends on its assumed size, with Monte Carlo standard errors,
     and whether "fix conditions at source" still beats gentle within-person weighting.

Run:  python3 sensitivity.py sensitivity_results.json      (about 10 minutes on a laptop)
Every number is a property of the simulated design under stated assumptions, not evidence about people.
"""
import json, sys
import numpy as np
from scipy import stats
import h2sim as h2

OUT = {}

# ---------- A. power by spread ----------
def power(N, d, s, a=.05):
    df = N - 1; nc = d / (s / np.sqrt(N)); tc = stats.t.ppf(1 - a, df)
    return float(1 - stats.nct.cdf(tc, df, nc))

def n_for(d, s, target=.8, lo=60, hi=100):
    for N in range(lo, hi + 1):
        if power(N, d, s) >= target:
            return N
    return hi

A = {}
for s in (6.0, 8.5, 11.0, 13.0):
    A[str(s)] = {f"N={N}": round(power(N, 3.0, s), 3) for N in (60, 80, 100)}
    A[str(s)]["N for 80% (3 pp)"] = next(N for N in range(10, 1000) if power(N, 3.0, s) >= .8)
    A[str(s)]["d = 3/SD"] = round(3.0 / s, 2)
    A[str(s)]["MDE at N=60, 80% power (pp)"] = round(float(
        next(x for x in np.arange(.5, 20, .05) if power(60, x, s) >= .8)), 1)
OUT["A_power_by_spread"] = A

# ---------- B. adaptive rule ----------
# Person-level D_i ~ Normal(mu, sd) in percentage points. After 20 people a blinded analyst receives only
# the SD of D_i and sets N = smallest N in [60, 100] giving 80% power for 3 pp (100 if none does).
# The primary test is then a one-sided one-sample t-test on all N people.
def adaptive(mu, sd, reps, rng):
    sig = conf = 0; Ns = []
    for _ in range(reps):
        first = rng.normal(mu, sd, 20)
        N = n_for(3.0, first.std(ddof=1))
        x = np.concatenate([first, rng.normal(mu, sd, N - 20)])
        p = stats.ttest_1samp(x, 0, alternative="greater").pvalue
        sig += p < .05; conf += (p < .05) and x.mean() >= 3; Ns.append(N)
    return dict(sig=sig / reps, sig_and_ge3=conf / reps, mean_N=float(np.mean(Ns)),
                share_N100=float(np.mean(np.array(Ns) == 100)), mcse=float(np.sqrt(sig / reps * (1 - sig / reps) / reps)))

rng = np.random.default_rng(20261112)
B = {}
for sd in (6.0, 8.5, 11.0):
    for mu in (0.0, 3.0, 3.5, 6.0):
        B[f"sd={sd}|mu={mu}"] = adaptive(mu, sd, 20000 if mu == 0 else 5000, rng)
        # fixed N=60 comparator
    x = rng.normal(0, sd, (20000, 60))
    B[f"sd={sd}|mu=0.0|fixed N=60"] = dict(sig=float((stats.ttest_1samp(x, 0, axis=1, alternative='greater').pvalue < .05).mean()))
OUT["B_adaptive_rule"] = B

# ---------- C. side bias ----------
# 48 items; reflective attractor rate 50%; under pressure each person gains a bias b_i toward the left
# option (b_i ~ U(0, .25)). The attractor's side is randomised per item per person (fair coin).
def side_bias(reps, rng, N=60, J=48, randomise=True):
    sig = 0; Dm = []
    for _ in range(reps):
        b = rng.uniform(0, .25, N)[:, None]
        left_is_attr = rng.random((N, J)) < .5 if randomise else np.ones((N, J), bool)
        pR = .5
        p_left_P = .5 + b
        pP = np.where(left_is_attr, p_left_P, 1 - p_left_P)
        yR = rng.random((N, J)) < pR; yP = rng.random((N, J)) < pP
        D = (yP.mean(1) - yR.mean(1)) * 100
        sig += stats.ttest_1samp(D, 0, alternative="greater").pvalue < .05; Dm.append(D.mean())
    return dict(false_alarm=sig / reps, mean_D=float(np.mean(Dm)))
OUT["C_side_bias"] = {"attractor side randomised": side_bias(4000, rng),
                      "attractor always on the left (design error, for contrast)": side_bias(1000, rng, randomise=False)}

# ---------- D. H2 source-fix sensitivity ----------
def h2_rep(sc, seed):
    return h2.evaluate(h2.simulate(sc, np.random.default_rng(seed)))

base_sc = dict(h2.SC["Pull; moderate proxy"])
REPS = 400
base = [h2_rep(base_sc, 50000 + k) for k in range(REPS)]
uB = np.array([r["Uniform majority"]["B"] for r in base])
gentle = np.array([r["CWHF (within-person score)"]["B"] for r in base])
sharp = np.array([r["CWHF within-person, sharp weights (c^4)"]["B"] for r in base])
sharp_ess = np.mean([r["CWHF within-person, sharp weights (c^4)"]["ess"] for r in base])
D = {"baseline_lean_pp": dict(mean=float(uB.mean() * 100), mcse=float(uB.std(ddof=1) / np.sqrt(REPS) * 100)),
     "gentle_within_person_removed": float(1 - gentle.mean() / uB.mean()),
     "sharp_within_person_removed": float(1 - sharp.mean() / uB.mean()), "sharp_within_person_ess": float(sharp_ess)}
for shift in (.05, .10, .15, .20):
    sc = dict(base_sc, z_shift=-shift)
    fx = np.array([h2_rep(sc, 90000 + int(shift * 1000) * 1000 + k)["Uniform majority"]["B"] for k in range(REPS)])
    red = 1 - fx.mean() / uB.mean()
    se = float(np.sqrt((fx.std(ddof=1) / fx.mean()) ** 2 + (uB.std(ddof=1) / uB.mean()) ** 2) / np.sqrt(REPS) * (1 - red))
    D[f"source fix, depletion lowered by {shift:.2f} (0-1 scale)"] = dict(
        removed=float(red), mcse=se, beats_gentle=bool(red > D["gentle_within_person_removed"]),
        beats_sharp=bool(red > D["sharp_within_person_removed"]))
OUT["D_h2_source_fix"] = D

json.dump(OUT, open(sys.argv[1] if len(sys.argv) > 1 else "sensitivity_results.json", "w"), indent=1)
print(json.dumps(OUT, indent=1))
