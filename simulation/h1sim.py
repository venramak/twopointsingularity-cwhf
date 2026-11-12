"""
H1 design simulation — flip-asymmetry study (CWHF v2.0).

Purpose: test the EXPERIMENT, not the idea. Simulated people are generated
with known truth (pull present or absent), the study is run on them, and the
pre-registered analyses are checked for
  - false-positive rate when there is no directional pull (Type I error)
  - power when there is a pull of a given size
  - bias from known confounds (noise-induced regression, timer artefact on
    long answers, carry-over between rounds, order, timeouts)
Nothing here is evidence that the real effect exists.

Model (per participant i, item j, condition R = reflective, P = pressured):
  logit P_R(attr) = b_i + e_j + a_i * clar_j * C_j
  logit P_P(attr) = b_i + e_j + (1 - att_i) * a_i * clar_j * C_j + pull_i,t
  P_P = (1 - lapse_i) * sigmoid(.) + lapse_i / 2
  C_j = +1 if the attractor option is the correct one, -1 otherwise (balanced)
Length items under a timer: with prob s_ij the long option cannot be read;
then the long option is chosen with prob q (q < .5 = "pick what you read").
"""
import numpy as np, json, sys, time

rng_master = np.random.default_rng(20261111)
sig = lambda x: 1 / (1 + np.exp(-x))

# ---------------- personas ----------------
# share, skill a, lapse, attenuation, susceptibility multiplier, reading words/s, style pref b
PERSONAS = {
 "Rested domain specialist": dict(share=.10, a=2.2, lapse=.02, att=.15, sus=.6, rs=5.0, b=0.0),
 "Gig annotator paid per item": dict(share=.18, a=1.3, lapse=.07, att=.30, sus=1.2, rs=5.5, b=0.1),
 "Night-shift health worker": dict(share=.12, a=1.4, lapse=.09, att=.40, sus=1.3, rs=4.0, b=0.0),
 "Second-language reader": dict(share=.15, a=1.3, lapse=.05, att=.30, sus=1.0, rs=2.8, b=0.0),
 "University student": dict(share=.15, a=1.5, lapse=.05, att=.25, sus=1.0, rs=4.5, b=0.0),
 "Retiree": dict(share=.10, a=1.6, lapse=.04, att=.25, sus=0.9, rs=3.2, b=-0.1),
 "Overconfident generalist": dict(share=.10, a=1.0, lapse=.06, att=.30, sus=1.0, rs=4.5, b=0.35),
 "Anxious perfectionist": dict(share=.10, a=1.8, lapse=.10, att=.35, sus=0.7, rs=3.8, b=-0.15),
}
PNAMES = list(PERSONAS)
SHARES = np.array([PERSONAS[p]["share"] for p in PNAMES]); SHARES /= SHARES.sum()

def draw_people(n, rng):
    k = rng.choice(len(PNAMES), size=n, p=SHARES)
    P = [PERSONAS[PNAMES[x]] for x in k]
    g = lambda key, sd, lo=None, hi=None: np.clip(np.array([p[key] for p in P]) + rng.normal(0, sd, n), lo, hi)
    return dict(persona=k, a=g("a", .4, .2, None), lapse=g("lapse", .02, 0, .3), att=g("att", .08, 0, .8),
                sus=g("sus", .25, 0, None), rs=g("rs", .6, 1.5, None), b=g("b", .2))

def make_items(K, rng, types=(0, 1, 2)):
    t = np.array([types[i % len(types)] for i in range(K)])           # 0 confident, 1 agreeable, 2 longer
    C = np.where((np.arange(K) // len(types)) % 2 == 0, 1, -1)          # attractor correct in half of each type
    clar = np.clip(rng.normal(1.0, .25, K), .4, 1.6)                     # item clarity
    e = rng.normal(0, .3, K)                                             # item-specific attractiveness
    words = np.where(t == 2, rng.integers(60, 90, K) + rng.integers(8, 15, K), rng.integers(30, 60, K))  # total words in the pair
    longw = np.where(t == 2, words - rng.integers(8, 15, K), words / 2)
    return dict(t=t, C=C, clar=clar, e=e, words=words, longw=longw)

def run_study(N, K, sc, rng):
    """Simulate one study; return per-observation arrays."""
    ppl = draw_people(N, rng); it = make_items(K, rng, sc.get("types", (0, 1, 2)))
    a, b, lapse, att, sus, rs = (ppl[k][:, None] for k in ("a", "b", "lapse", "att", "sus", "rs"))
    t, C, clar, e = it["t"][None], it["C"][None], it["clar"][None], it["e"][None]
    # directional pull (logit) by attractor type; heterogeneous across people
    mu = np.array(sc["pull"])[t]                                   # shape (1,K) via type
    pull = mu * sus + rng.normal(0, sc.get("pull_sd", .15), (N, 1)) * (np.array(sc["pull"]).max() > 0)
    bias_b = b + np.array(sc.get("b_type", (0, 0, 0)))[t]          # reflective style preference by type
    etaR = bias_b + e + a * clar * C
    etaP = bias_b + e + (1 - att) * a * clar * C + pull
    pR = sig(etaR)
    pP = (1 - lapse) * sig(etaP) + lapse / 2
    pP0 = (1 - lapse) * sig(etaP - pull) + lapse / 2              # same person, no pull (counterfactual)
    # timer: per-item limit
    if sc["timer"] == "fixed":
        T = np.full((1, K), 7.0)
    else:  # scaled to length: 0.12 s per word, minimum 5 s  (about half normal reading time)
        T = np.maximum(5.0, 0.12 * it["words"][None])
    need = it["words"][None] / (rs * 1.6)                          # skim reading time under pressure
    p_to = sig((need - 2.0 * T) / 0.8)                             # timeout probability
    longneed = it["longw"][None] / (rs * 1.6)
    s = np.where(t == 2, sig((longneed - .7 * T) / 1.0), 0.0)      # cannot read long option
    q = sc.get("q", .5)
    pP = (1 - s) * pP + s * q
    pP0 = (1 - s) * pP0 + s * q
    prim = np.isin(np.broadcast_to(t, (N, K)), (0, 1))
    true_pull_primary = float((pP - pP0)[prim].mean())
    true_dpp_primary = float((pP - pR)[prim].mean())
    # draw
    order_R_first = rng.random((N, 1)) < .5
    first_is_R = np.broadcast_to(order_R_first, (N, K))
    yR = rng.random((N, K)) < pR
    yP = rng.random((N, K)) < pP
    timeout = rng.random((N, K)) < p_to
    rho = sc.get("carry", .15)
    copy = rng.random((N, K)) < rho
    # second-round answer copies the first with prob rho
    yP = np.where(first_is_R & copy, yR, yP)
    yR = np.where(~first_is_R & copy & ~timeout, yP, yR)
    return dict(yR=yR, yP=yP, timeout=timeout, C=np.broadcast_to(C, (N, K)), t=np.broadcast_to(t, (N, K)),
                persona=ppl["persona"], true_dpp=(pP - pR).mean(), true_pull=true_pull_primary, true_dpp_primary=true_dpp_primary)

def analyse(d, types=None):
    yR, yP, to, C, t = d["yR"], d["yP"], d["timeout"], d["C"], d["t"]
    N, K = yR.shape
    keep = ~to
    if types is not None:
        keep = keep & np.isin(t, types)
    # ---- flips ----
    toward = (keep & yP & ~yR).sum(1).astype(float)
    away = (keep & ~yP & yR).sum(1).astype(float)
    nflip = toward + away
    tot = nflip.sum()
    pi = toward.sum() / tot if tot else np.nan
    # naive pooled binomial (ignores clustering) — what the v1.0 demo logic would do if pooled
    from scipy.stats import binomtest
    p_naive = binomtest(int(toward.sum()), int(tot), .5, alternative="greater").pvalue if tot else 1
    # cluster-robust ratio estimator for pi
    m = N
    resid = toward - pi * nflip
    se_pi = np.sqrt((resid ** 2).sum() * m / (m - 1)) / tot if tot else np.nan
    z_pi = (pi - .5) / se_pi if tot else 0
    # ---- attenuation-corrected pull (linear probability model, cluster-robust) ----
    rows = []
    for cond, y in ((0, yR), (1, yP)):
        k = keep.copy() if cond == 1 else (keep if True else None)
        rows.append((np.where(k)[0], y[k], np.full(k.sum(), cond), C[k]))
    gid = np.concatenate([r[0] for r in rows]); Y = np.concatenate([r[1] for r in rows]).astype(float)
    Pv = np.concatenate([r[2] for r in rows]).astype(float); Cv = np.concatenate([r[3] for r in rows]).astype(float)
    X = np.column_stack([np.ones_like(Y), Pv, Cv, Pv * Cv])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ Y
    u = Y - X @ beta
    G = np.zeros((N, 4))
    np.add.at(G, gid, X * u[:, None])
    meat = G.T @ G
    V = XtX_inv @ meat @ XtX_inv * (N / (N - 1))
    b0, bP, bC, bPC = beta
    bprime = b0 - .5
    delta = bP - (bPC / bC) * bprime
    grad = np.array([-bPC / bC, 1.0, bPC * bprime / bC ** 2, -bprime / bC])
    se_d = float(np.sqrt(grad @ V @ grad))
    naive_diff = bP  # uncorrected pressured-minus-reflective attractor rate
    se_naive = float(np.sqrt(V[1, 1]))
    flip_rate = tot / keep.sum()
    return dict(pi=pi, p_naive=p_naive, z_pi=z_pi, delta=delta, se_d=se_d, naive_diff=naive_diff,
                se_naive=se_naive, flip_rate=flip_rate, timeout_rate=to.mean(), acc_drop=None)

def decide(r):
    return dict(
        old_rule_pi060=r["pi"] >= .60,
        naive_binomial=r["p_naive"] < .05,
        cluster_pi=r["z_pi"] > 1.645,
        naive_diff=(r["naive_diff"] / r["se_naive"]) > 1.645,
        corrected_delta=(r["delta"] / r["se_d"]) > 1.645,
        corrected_delta_sesoi=((r["delta"] / r["se_d"]) > 1.645) and r["delta"] >= .03,
    )

SCEN = {
 # nulls
 "Null: noise only, balanced design": dict(pull=(0, 0, 0), timer="scaled", carry=.15),
 "Null: noise + reflective dislike of attractors": dict(pull=(0, 0, 0), timer="scaled", carry=.15, b_type=(-.5, -.5, -.5)),
 "Null: fixed 7 s timer, long options unread": dict(pull=(0, 0, 0), timer="fixed", carry=.15, q=.25),
 # effects
 "Effect: small pull (logit 0.20)": dict(pull=(.2, .2, .2), timer="scaled", carry=.15),
 "Effect: moderate pull (logit 0.35)": dict(pull=(.35, .35, .35), timer="scaled", carry=.15),
 "Effect: moderate pull, same-sitting carry-over": dict(pull=(.35, .35, .35), timer="scaled", carry=.6),
 "Effect: moderate pull, fixed timer on long items": dict(pull=(.35, .35, .35), timer="fixed", carry=.15, q=.25),
}

def run(name, N, K, reps, types=None, seed=0):
    rng = np.random.default_rng(seed)
    sc = SCEN[name]
    out = {k: 0 for k in decide(dict(pi=.5, p_naive=1, z_pi=0, delta=0, se_d=1, naive_diff=0, se_naive=1))}
    agg = dict(pi=[], delta=[], naive=[], flip=[], dpp=[], to=[], tpull=[], se=[])
    for r_ in range(reps):
        d = run_study(N, K, sc, rng)
        r = analyse(d, types)
        for k, v in decide(r).items(): out[k] += v
        agg["pi"].append(r["pi"]); agg["delta"].append(r["delta"]); agg["naive"].append(r["naive_diff"])
        agg["flip"].append(r["flip_rate"]); agg["dpp"].append(d["true_dpp"]); agg["to"].append(r["timeout_rate"]); agg["tpull"].append(d["true_pull"]); agg["se"].append(r["se_d"])
    res = {k: v / reps for k, v in out.items()}
    res.update({f"mean_{k}": float(np.nanmean(v)) for k, v in agg.items()})
    return res

if __name__ == "__main__":
    t0 = time.time()
    results = {}
    REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    # A. operating characteristics at the proposed design: N=60, K=48 (confident+agreeable primary)
    for name in SCEN:
        results[f"A|{name}|all types"] = run(name, 60, 72, REPS, None, seed=1)
        results[f"A|{name}|primary (confident+agreeable)"] = run(name, 60, 72, REPS, (0, 1), seed=1)
    # B. power curve for the corrected test on primary items, by N and K
    for name in ["Effect: small pull (logit 0.20)", "Effect: moderate pull (logit 0.35)", "Null: noise + reflective dislike of attractors"]:
        for N in (20, 30, 40, 60, 80, 120):
            for K in (48, 72, 96):
                results[f"B|{name}|N={N}|K={K}"] = run(name, N, K, REPS, (0, 1), seed=2)
    json.dump(results, open(sys.argv[2] if len(sys.argv) > 2 else "h1_results.json", "w"), indent=1)
    print("done", round(time.time() - t0, 1), "s")


# ---------------------------------------------------------------------------
# Estimator v2: flip-level logistic model with baseline adjustment and
# two-way (participant x item) cluster-robust errors, t reference with N-1 df.
# Model, on discordant pairs only:
#   logit P(flip goes toward attractor) = g0 + g1*C_j + g2*(reflective attractor rate_i - .5)
# H1: g0 > 0  (pull toward the attractor for a person whose reflective
# baseline is balanced, averaged over correct/incorrect attractors).
# ---------------------------------------------------------------------------
def _logit_fit(X, y, iters=25):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = sig(X @ b); W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + 1e-8 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p)); b += step
        if np.abs(step).max() < 1e-8: break
    p = sig(X @ b); W = p * (1 - p)
    Hinv = np.linalg.inv(X.T @ (X * W[:, None]))
    return b, p, Hinv

def _cluster_meat(S, g, ng):
    G = np.zeros((ng, S.shape[1])); np.add.at(G, g, S); return G.T @ G

def analyse_v2(d, types=(0, 1)):
    from scipy.stats import t as tdist
    yR, yP, to, C, t = d["yR"], d["yP"], d["timeout"], d["C"], d["t"]
    N, K = yR.shape
    keep = (~to) & np.isin(t, types)
    base = np.array([yR[i][keep[i]].mean() if keep[i].any() else .5 for i in range(N)]) - .5
    flip = keep & (yP != yR)
    pi_idx, it_idx = np.where(flip)
    y = yP[flip].astype(float)                     # 1 = flip toward attractor
    X = np.column_stack([np.ones(len(y)), C[flip].astype(float), base[pi_idx]])
    if len(y) < 10: return dict(g0=np.nan, z=0, se=np.nan, pi=np.nan)
    b, p, Hinv = _logit_fit(X, y)
    S = X * (y - p)[:, None]
    M = _cluster_meat(S, pi_idx, N) + _cluster_meat(S, it_idx, K) - S.T @ S   # two-way cluster
    V = Hinv @ M @ Hinv * (N / (N - 1))
    se = float(np.sqrt(max(V[0, 0], 1e-12)))
    tval = b[0] / se
    crit = tdist.ppf(.95, N - 1)
    # probability-scale pull implied by g0 among flips is not the population pull; report pi too
    return dict(g0=float(b[0]), se=se, z=float(tval), crit=float(crit), pi=float(y.mean()))

SCEN["Null: noise + reflective liking of attractors"] = dict(pull=(0, 0, 0), timer="scaled", carry=.15, b_type=(.5, .5, .5))
SCEN["Null: high lapse (very tired sample)"] = dict(pull=(0, 0, 0), timer="scaled", carry=.15, lapse_add=.10)

_old_run_study = run_study
def run_study(N, K, sc, rng):
    if sc.get("lapse_add"):
        global PERSONAS
    return _old_run_study(N, K, sc, rng)

def run_v2(name, N, K, reps, seed=0, types=(0, 1)):
    rng = np.random.default_rng(seed); sc = SCEN[name]
    hits = dict(v2=0, v1_corrected=0, cluster_pi=0, old_pi060=0); g0s = []; tp = []; pis = []
    for _ in range(reps):
        d = run_study(N, K, sc, rng)
        r2 = analyse_v2(d, types); r1 = analyse(d, types)
        hits["v2"] += r2["z"] > r2.get("crit", 1.7)
        hits["v1_corrected"] += (r1["delta"] / r1["se_d"]) > 1.645
        hits["cluster_pi"] += r1["z_pi"] > 1.645
        hits["old_pi060"] += r1["pi"] >= .60
        g0s.append(r2["g0"]); tp.append(d["true_pull"]); pis.append(r1["pi"])
    out = {k: v / reps for k, v in hits.items()}
    out.update(mean_g0=float(np.nanmean(g0s)), mean_true_pull=float(np.mean(tp)), mean_pi=float(np.nanmean(pis)))
    return out


# ---------------------------------------------------------------------------
# FINAL pre-registered estimator (v3)
# Estimand  D = mean over people and balanced items of
#               P(attractor | pressured) - P(attractor | reflective)
# Fit: linear probability model  y ~ 1 + Pressure + C + Pressure:C  on primary
# (length-matched confident + agreeable) items; D = coefficient on Pressure
# (C is coded +-1 and balanced, so this is the average over correct and
# incorrect attractors). SEs: two-way cluster-robust (participant, item),
# small-sample factor, t reference with N-1 df. One-sided alpha = .05.
# Confirmation rule: one-sided test passes AND D >= SESOI (3 percentage points).
# ---------------------------------------------------------------------------
from scipy.stats import t as _tdist
SESOI = 0.03
def analyse_v3(d, types=(0, 1)):
    yR, yP, to, C, t = d["yR"], d["yP"], d["timeout"], d["C"], d["t"]
    N, K = yR.shape
    keep = (~to) & np.isin(t, types)
    pid = np.repeat(np.arange(N)[:, None], K, 1)[keep]
    iid = np.repeat(np.arange(K)[None], N, 0)[keep]
    Cv = C[keep].astype(float)
    Y = np.concatenate([yR[keep], yP[keep]]).astype(float)
    Pv = np.concatenate([np.zeros(keep.sum()), np.ones(keep.sum())])
    Cc = np.concatenate([Cv, Cv]); g1 = np.concatenate([pid, pid]); g2 = np.concatenate([iid, iid])
    X = np.column_stack([np.ones_like(Y), Pv, Cc, Pv * Cc])
    XtXi = np.linalg.inv(X.T @ X); beta = XtXi @ X.T @ Y; u = Y - X @ beta
    S = X * u[:, None]
    # intersection cluster = (participant, item): both rows of the same pair
    g12 = g1 * K + g2
    _, g12c = np.unique(g12, return_inverse=True)
    M = _cluster_meat(S, g1, N) + _cluster_meat(S, g2, K) - _cluster_meat(S, g12c, g12c.max() + 1)
    V = XtXi @ M @ XtXi * (min(N, K) / (min(N, K) - 1))
    D = float(beta[1]); se = float(np.sqrt(max(V[1, 1], 1e-12)))
    tval = D / se; crit = float(_tdist.ppf(.95, N - 1))
    toward = (keep & yP & ~yR).sum(); away = (keep & ~yP & yR).sum()
    pi = toward / (toward + away) if toward + away else np.nan
    acc_R = np.where(C[keep] == 1, yR[keep], ~yR[keep]).mean(); acc_P = np.where(C[keep] == 1, yP[keep], ~yP[keep]).mean()
    return dict(D=D, se=se, t=tval, crit=crit, sig=tval > crit, confirm=(tval > crit) and D >= SESOI,
                lo90=D - crit * se, pi=float(pi), flip_rate=float((toward + away) / keep.sum()),
                acc_drop=float(acc_R - acc_P), interaction=float(beta[3]))

def run_v3(name, N, K, reps, seed=0, types=(0, 1)):
    rng = np.random.default_rng(seed); sc = SCEN[name]
    acc = dict(sig=0, confirm=0, old_pi060=0, cover=0); Ds = []; tps = []; pis = []; fl = []; drops = []
    for _ in range(reps):
        d = run_study(N, K, sc, rng); r = analyse_v3(d, types)
        acc["sig"] += r["sig"]; acc["confirm"] += r["confirm"]; acc["old_pi060"] += r["pi"] >= .60
        Ds.append(r["D"]); tps.append(d["true_dpp_primary"]); pis.append(r["pi"]); fl.append(r["flip_rate"]); drops.append(r["acc_drop"])
        acc["cover"] += abs(r["D"] - d["true_dpp_primary"]) <= 1.96 * r["se"]
    out = {k: v / reps for k, v in acc.items()}
    out.update(mean_D=float(np.mean(Ds)), true_D=float(np.mean(tps)), sd_D=float(np.std(Ds)),
               mean_pi=float(np.nanmean(pis)), flip_rate=float(np.mean(fl)), acc_drop=float(np.mean(drops)))
    return out
