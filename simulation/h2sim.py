"""
H2 design simulation — does state weighting beat existing rater-quality methods?

Naturalistic labelling: no manipulation. Raters label items across several
sessions; each session has a depletion state z in [0,1]. When depleted,
choices drift toward the attractor (only if a pull exists), lapse more, and
discriminate less. A session score c is built from a two-item self-report and
response-time variability, both noisy and partly biased by persona.

Ground truth for H2 is the reflective population label of each item.
Methods compared:
  uniform majority | attention-check filter | Dawid-Skene (one-coin) |
  CWHF raw c | CWHF within-person c | Dawid-Skene x CWHF within-person | oracle (true state)
"""
import numpy as np, json, sys
sig = lambda x: 1 / (1 + np.exp(-x))

# share, skill, lapse, attenuation, susceptibility, mean depletion, self-report bias, language/RT penalty
PERS = {
 "Rested domain specialist":   (.10, 2.2, .02, .15, .6, .20,  .00, .0),
 "Gig annotator paid per item":(.18, 1.3, .07, .30, 1.2, .45, .10, .0),
 "Night-shift health worker":  (.12, 1.4, .09, .40, 1.3, .65, .00, .0),
 "Second-language reader":     (.15, 1.3, .05, .30, 1.0, .40, .00, 1.0),
 "University student":         (.15, 1.5, .05, .25, 1.0, .40, .00, .0),
 "Retiree":                    (.10, 1.6, .04, .25, .9, .30, .00, .5),
 "Overconfident generalist":   (.10, 1.0, .06, .30, 1.0, .40, .15, .0),
 "Anxious perfectionist":      (.10, 1.8, .10, .35, .7, .45, -.20, .0),
}
NAMES = list(PERS); SH = np.array([v[0] for v in PERS.values()]); SH /= SH.sum()
L2 = NAMES.index("Second-language reader"); NIGHT = NAMES.index("Night-shift health worker")

def simulate(sc, rng, M=60, J=600, r=5, sess_len=10):
    k = rng.choice(len(NAMES), M, p=SH)
    P = np.array([PERS[NAMES[x]][1:] for x in k])
    a = np.clip(P[:, 0] + rng.normal(0, .4, M), .2, None); lapse = np.clip(P[:, 1] + rng.normal(0, .02, M), 0, .3)
    att = np.clip(P[:, 2] + rng.normal(0, .08, M), 0, .8); sus = np.clip(P[:, 3] + rng.normal(0, .25, M), 0, None)
    zmean = P[:, 4]; srb = P[:, 5]; lang = P[:, 6]; b = rng.normal(0, .2, M)
    # items
    C = np.where(np.arange(J) % 2 == 0, 1, -1); rng.shuffle(C)
    clar = np.clip(rng.normal(1, .25, J), .4, 1.6); e = rng.normal(0, .3, J)
    # reflective truth: population-majority reflective choice
    pop = rng.choice(len(NAMES), 4000, p=SH)
    pa = np.array([PERS[NAMES[x]][2] for x in pop]) + rng.normal(0, .4, 4000)
    pb = rng.normal(0, .2, 4000)
    pR_pop = sig(pb[:, None] + e[None] + np.clip(pa, .2, None)[:, None] * clar[None] * C[None]).mean(0)
    truth = (pR_pop > .5).astype(int)     # 1 = attractor is the reflective label
    # assignment: each item gets r distinct raters
    loads = np.zeros(M, int); assign = []
    for j in range(J):
        cand = np.argsort(loads + rng.random(M) * .5)[:r]; loads[cand] += 1; assign.append(cand)
    assign = np.array(assign)
    # queue: optionally items in a "night queue" share elevated depletion (correlated state)
    night_items = rng.random(J) < (.5 if sc.get("correlated") else 0)
    # sessions: per rater, items in order, chunked
    lab = np.zeros((J, r), int); W = {}
    sess_id = np.zeros((J, r), int); att_fail = np.zeros((J, r), bool)
    zs = {}; srs = {}; rtvs = {}
    rows = []
    for i in range(M):
        jj, slot = np.where(assign == i)
        order = rng.permutation(len(jj))
        jj, slot = jj[order], slot[order]
        # sort so that night items cluster into the same sessions
        o2 = np.argsort(night_items[jj], kind="stable"); jj, slot = jj[o2], slot[o2]
        for s0 in range(0, len(jj), sess_len):
            js, ss = jj[s0:s0 + sess_len], slot[s0:s0 + sess_len]
            night = night_items[js].mean() > .5
            z = np.clip(rng.normal(zmean[i] + (.30 if night else 0) + sc.get('z_shift', 0), .18), 0, 1)
            sid = (i, s0)
            zs[sid] = z
            # proxies
            srs[sid] = np.clip(1 - z + srb[i] + rng.normal(0, sc["sr_noise"]), 0, 1)
            rtvs[sid] = np.clip(.6 * z + .4 * lang[i] * sc.get("lang_rt", 1) + rng.normal(0, sc["rt_noise"]), 0, 1)
            pull = sc["pull"] * sus[i] * (4 * max(0.0, z - .5) * 2 if sc.get('threshold') else 2 * z)
            lap = np.clip(lapse[i] * (.5 + z), 0, .5)
            atti = np.clip(att[i] * 2 * z, 0, .95)
            eta = b[i] + e[js] + (1 - atti) * a[i] * clar[js] * C[js] + pull
            p = (1 - lap) * sig(eta) + lap / 2
            y = (rng.random(len(js)) < p).astype(int)
            fail = rng.random() < min(.9, lap * 3)
            for jx, sx, yx in zip(js, ss, y):
                lab[jx, sx] = yx; att_fail[jx, sx] = fail
                rows.append((jx, sx, i, sid))
    # build per-label session state arrays
    keymap = {}
    Z = np.zeros((J, r)); SR = np.zeros((J, r)); RT = np.zeros((J, r)); RID = np.zeros((J, r), int)
    for jx, sx, i, sid in rows:
        Z[jx, sx] = zs[sid]; SR[jx, sx] = srs[sid]; RT[jx, sx] = rtvs[sid]; RID[jx, sx] = i
    c_raw = np.clip(.5 * SR + .5 * (1 - RT), 0, 1)
    # within-person: subtract rater mean of c, add grand mean
    cm = np.zeros(M); cn = np.zeros(M)
    np.add.at(cm, RID.ravel(), c_raw.ravel()); np.add.at(cn, RID.ravel(), 1)
    c_wp = np.clip(c_raw - (cm / cn)[RID] + c_raw.mean(), 0, 1)
    return dict(lab=lab, truth=truth, pRpop=pR_pop, C=C, Z=Z, c_raw=c_raw, c_wp=c_wp, RID=RID, att_fail=att_fail,
                persona=k, M=M, night=night_items)

def wvote(lab, w):
    s = (w * (2 * lab - 1)).sum(1)
    tie = s == 0
    out = (s > 0).astype(int)
    out[tie] = (lab[tie].mean(1) > .5).astype(int)
    return out

def dawid_skene(lab, RID, M, iters=30):
    est = (lab.mean(1) > .5).astype(int)
    acc = np.full(M, .75)
    for _ in range(iters):
        corr = (lab == est[:, None]).astype(float)
        num = np.zeros(M); den = np.zeros(M)
        np.add.at(num, RID.ravel(), corr.ravel()); np.add.at(den, RID.ravel(), 1)
        acc = np.clip((num + 1) / (den + 2), .51, .99)
        w = np.log(acc / (1 - acc))[RID]
        est = wvote(lab, w)
    return np.log(acc / (1 - acc))[RID]

cw = lambda c: .10 + .90 * c ** 1.5

def evaluate(d):
    lab, truth = d["lab"], d["truth"]
    ones = np.ones_like(lab, float)
    ds = dawid_skene(lab, d["RID"], d["M"])
    methods = {
        "Uniform majority": ones,
        "Attention-check filter": np.where(d["att_fail"], 1e-6, 1.0),
        "Dawid-Skene reliability": ds,
        "CWHF (raw session score)": cw(d["c_raw"]),
        "CWHF (within-person score)": cw(d["c_wp"]),
        "Dawid-Skene x CWHF within-person": ds * cw(d["c_wp"]),
        "CWHF within-person, sharp weights (c^4)": .02 + .98 * d["c_wp"] ** 4,
        "CWHF raw, sharp weights (c^4)": .02 + .98 * d["c_raw"] ** 4,
        "Oracle (true state)": cw(1 - d["Z"]),
        "Oracle, sharp (true state, ^4)": .02 + .98 * (1 - d["Z"]) ** 4,
    }
    res = {}
    attr_wrong = d["C"] == -1
    l2 = d["persona"][d["RID"]] == L2
    night = d["persona"][d["RID"]] == NIGHT
    for name, w in methods.items():
        y = wvote(lab, w)
        dev = lab - d["pRpop"][:, None]                       # each label's deviation from the reflective population rate
        B = float((w * dev).sum() / w.sum())                  # directional lean of the (weighted) training signal
        ess = float(w.sum() ** 2 / (w ** 2).sum() / w.size)    # effective sample size as share of labels
        res[name] = dict(B=B, ess=ess, acc=float((y == truth).mean()),
                         attr_err=float((y[attr_wrong] == 1).mean()),
                         w_L2_ratio=float(w[l2].mean() / w[~l2].mean()) if l2.any() else np.nan,
                         w_night_ratio=float(w[night].mean() / w[~night].mean()) if night.any() else np.nan)
    res["_proxy_validity"] = float(np.corrcoef(d["c_raw"].ravel(), -d["Z"].ravel())[0, 1])
    res["_proxy_validity_wp"] = float(np.corrcoef(d["c_wp"].ravel(), -d["Z"].ravel())[0, 1])
    return res

SC = {
 "No pull; valid proxy": dict(pull=0, sr_noise=.15, rt_noise=.10),
 "Pull; weak proxy": dict(pull=.35, sr_noise=.45, rt_noise=.35),
 "Pull; moderate proxy": dict(pull=.35, sr_noise=.25, rt_noise=.18),
 "Pull; strong proxy": dict(pull=.35, sr_noise=.12, rt_noise=.08),
 "Pull; moderate proxy; correlated state (night queue)": dict(pull=.35, sr_noise=.25, rt_noise=.18, correlated=True),
 "Pull; strong proxy; correlated state (night queue)": dict(pull=.35, sr_noise=.12, rt_noise=.08, correlated=True),
 "Threshold pull (only when badly depleted); moderate proxy": dict(pull=.35, sr_noise=.25, rt_noise=.18, threshold=True),
 "Threshold pull; moderate proxy; correlated state": dict(pull=.35, sr_noise=.25, rt_noise=.18, threshold=True, correlated=True),
 "Source fix: shorter sessions and breaks (state -0.15); moderate proxy": dict(pull=.35, sr_noise=.25, rt_noise=.18, z_shift=-.15),
}

def run(name, reps, seed, r=5):
    rng = np.random.default_rng(seed); acc = {}
    gains = []; gains_ds = []
    for _ in range(reps):
        res = evaluate(simulate(SC[name], rng, r=r))
        baseB = min(abs(res[m]["B"]) for m in ("Uniform majority", "Attention-check filter", "Dawid-Skene reliability"))
        uB = res["Uniform majority"]["B"]
        gains.append((baseB - abs(res["CWHF (within-person score)"]["B"])))      # reduction in |lean| vs best baseline
        gains_ds.append((abs(uB) - abs(res["CWHF (within-person score)"]["B"])) / abs(uB) if abs(uB) > 1e-4 else 0)  # share of lean removed vs uniform
        for m, v in res.items():
            if m.startswith("_"):
                acc.setdefault(m, []).append(v)
            else:
                for kk, vv in v.items(): acc.setdefault((m, kk), []).append(vv)
    out = {}
    for key, v in acc.items():
        v = np.array(v, float)
        if isinstance(key, tuple):
            out.setdefault(key[0], {})[key[1]] = float(np.nanmean(v)); out[key[0]][key[1] + "_sd"] = float(np.nanstd(v))
        else:
            out[key] = float(np.mean(v))
    g = np.array(gains); gd = np.array(gains_ds)
    out["_lean_reduction_vs_best_baseline_pp"] = dict(mean=float(g.mean() * 100), sd=float(g.std() * 100), p_positive=float((g > 0).mean()))
    out["_share_of_lean_removed_vs_uniform"] = dict(mean=float(gd.mean()), sd=float(gd.std()), p_ge_30pct=float((gd >= .30).mean()))
    return out

if __name__ == "__main__":
    REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    allres = {}
    for r in (5,):
        for name in SC:
            allres[f"{name}|r={r}"] = run(name, REPS, seed=1000 * list(SC).index(name) + r, r=r)
            R = allres[f"{name}|r={r}"]
            print(name, r, {m: (round(v["B"]*100, 2), round(v["ess"], 2), round(v["w_L2_ratio"], 2)) for m, v in R.items() if isinstance(v, dict) and "B" in v}, R["_share_of_lean_removed_vs_uniform"], round(R["_proxy_validity_wp"], 2), flush=True)
    json.dump(allres, open(sys.argv[2] if len(sys.argv) > 2 else "h2_results.json", "w"), indent=1)
