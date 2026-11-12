import sys, json, numpy as np
sys.path.insert(0, sys.argv[1]); import h1sim as h
from scipy.stats import ttest_1samp
h.SCEN["Effect: tiny pull (logit 0.10)"] = dict(pull=(.1, .1, .1), timer="scaled", carry=.15)
def person_test(d, types=(0, 1)):
    keep = (~d["timeout"]) & np.isin(d["t"], types)
    Di = np.array([(d["yP"][i][keep[i]].mean() - d["yR"][i][keep[i]].mean()) if keep[i].sum() >= 10 else np.nan for i in range(len(keep))])
    Di = Di[~np.isnan(Di)]
    t = ttest_1samp(Di, 0, alternative="greater")
    return Di.mean(), t.pvalue, Di.std(ddof=1)
out = {}
for name, reps in [("Null: noise only, balanced design", 2000), ("Null: noise + reflective dislike of attractors", 500), ("Null: noise + reflective liking of attractors", 500),
                   ("Effect: tiny pull (logit 0.10)", 600), ("Effect: small pull (logit 0.20)", 600), ("Effect: moderate pull (logit 0.35)", 600)]:
    for N in ((20, 30, 40, 60, 80, 120) if "Null: noise only" in name or "Effect" in name else (60,)):
        rng = np.random.default_rng(N + reps); sig = conf = 0; Ds = []; sds = []
        r = reps if N == 60 else 400
        for _ in range(r):
            d = h.run_study(N, 72, h.SCEN[name], rng); m, p, sd = person_test(d)
            sig += p < .05; conf += (p < .05) and m >= .03; Ds.append(m); sds.append(sd)
        out[f"{name}|{N}"] = dict(sig=sig / r, confirm=conf / r, mean_D=float(np.mean(Ds)), sd_Di=float(np.mean(sds)), reps=r)
        print(name, N, round(sig / r, 3), round(conf / r, 3), round(np.mean(Ds), 4), round(np.mean(sds), 3), flush=True)
json.dump(out, open(sys.argv[2], "w"), indent=1)
