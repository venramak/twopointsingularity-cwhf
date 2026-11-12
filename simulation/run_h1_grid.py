import sys, json, numpy as np
sys.path.insert(0, sys.argv[1])
import h1sim as h
h.SCEN["Effect: tiny pull (logit 0.10)"] = dict(pull=(.1, .1, .1), timer="scaled", carry=.15)
out = {}
REPS = 400
for name in ["Null: noise only, balanced design", "Effect: tiny pull (logit 0.10)", "Effect: small pull (logit 0.20)", "Effect: moderate pull (logit 0.35)"]:
    for N in (20, 30, 40, 60, 80, 120):
        for K in (48, 72, 96):
            r = h.run_v3(name, N, K, REPS, seed=N * 7 + K)
            out[f"{name}|{N}|{K}"] = r
            print(name, N, K, round(r["sig"], 3), round(r["confirm"], 3), flush=True)
# operating characteristics table at proposed design N=60, K=72 (48 primary)
ops = {}
for name in h.SCEN:
    if "high lapse" in name: continue
    ops[name] = h.run_v3(name, 60, 72, 1000, seed=11)
# length items analysed on their own, under fixed vs scaled timer, null and effect
lens = {}
for name in ["Null: fixed 7 s timer, long options unread", "Null: noise only, balanced design", "Effect: moderate pull (logit 0.35)", "Effect: moderate pull, fixed timer on long items"]:
    lens[name] = h.run_v3(name, 60, 72, 500, seed=13, types=(2,))
# persona breakdown under moderate effect, one large study
rng = np.random.default_rng(99)
d = h.run_study(4000, 72, h.SCEN["Effect: moderate pull (logit 0.35)"], rng)
keep = (~d["timeout"]) & np.isin(d["t"], (0, 1))
pers = {}
for k, pname in enumerate(h.PNAMES):
    m = d["persona"] == k
    kk = keep[m]
    pers[pname] = dict(n=int(m.sum()), D=float(d["yP"][m][kk].mean() - d["yR"][m][kk].mean()),
                       timeout=float(d["timeout"][m].mean()))
json.dump(dict(grid=out, ops=ops, length=lens, personas=pers), open(sys.argv[2], "w"), indent=1)
