# Signal ≠ Value

**Does pressure bend the human feedback that trains AI?** Paper (version 1.1), technical supplement and companion lab (Venkat Rao, 2026). Preprint, not peer reviewed.

- **Paper (web):** https://cwhf.vercel.app/paper/
- **Paper (PDF):** [`paper/Signal-not-Value_Rao_2026.pdf`](paper/Signal-not-Value_Rao_2026.pdf)
- **Technical supplement (PDF):** [`paper/Signal-not-Value_Technical-Supplement.pdf`](paper/Signal-not-Value_Technical-Supplement.pdf): full protocol, item specification, pre-registration text for H1–H3
- **Lab (canonical):** https://twopointsingularity.com/lab/ · mirror: https://cwhf.vercel.app/
- **First paper:** [doi:10.5281/zenodo.19235970](https://doi.org/10.5281/zenodo.19235970) (March 2026). This paper supersedes its claims.
- **Earlier pre-registration:** [OSF W7NYH](https://osf.io/w7nyh). A registration of the revised H1–H3 is pending.

## The question

When people judge under pressure, do their choices move away from the choices the same people make on reflection, in a consistent direction rather than at random? The primary measure, **D**, is the signed reflective–pressured choice gap toward an "attractor" answer; the secondary measure, **M**, is the share of items where the two choices differ in either direction. If D is reliably above zero, collecting more human feedback entrenches the lean instead of averaging it away. The smallest effect of interest is 3 percentage points, and results are read from the confidence interval, not a single p-value.

The study tests condition-dependent preferences over answers. It does not identify moral values, authentic identity or anyone's correct preference.

## What is here

| Path | Contents |
| --- | --- |
| `paper/` | The paper (web page and A4 PDF) and the technical supplement (PDF). |
| `index.html` | The lab, v2.2: evidence base (71 cases), the Reach room on every digital signal, a flip-test demonstration (not research participation), the design simulation with a reproducibility receipt, the CWHF v2 proposal, method and the claims ledger. One static page; no cookies, storage or analytics; readable without JavaScript. |
| `simulation/` | Code and result files behind every simulated number in the lab and the paper. `sensitivity.py` adds power by spread, a check of the blinded sample-size rule, a side-bias check and the sensitivity of the source-fix result. |
| `schema/` | The step 1 conditions record as JSON Schema, with an example (paper, Appendix C). |
| `archive/` | Retired material kept for the record: whitepaper v2.0 and the CWHF Coherence Lab v0.4. Their claims are superseded. |

## Reproduce the simulation

Python 3.10+ with numpy and scipy:

```
cd simulation
python3 run_h1_grid.py . h1_results.json
python3 persontest.py . h1_person_test.json
python3 h2sim.py 80 h2_results.json
python3 power_sd.py
python3 sensitivity.py sensitivity_results.json   # about 3 minutes
```

Seeds are fixed by setting index, so reruns match. Every simulated result describes the study design under stated assumptions. None of it is evidence about how real people behave.

## Embedding

The lab may be embedded in an iframe on twopointsingularity.com (allowed by `vercel.json`):

```html
<iframe src="https://cwhf.vercel.app/" title="Signal ≠ Value Lab" loading="lazy" style="width:100%;height:85vh;border:0"></iframe>
```

## Licence and citation

Text, figures, paper, supplement, lab and data: CC BY 4.0 (`LICENSE`). Simulation code: Apache-2.0 (`LICENSE-CODE`). Cite with `CITATION.cff`, or:

Rao, V. (2026). *Signal ≠ Value: Does Pressure Bend the Human Feedback That Trains AI?* Preprint. Two Point Singularity. https://cwhf.vercel.app/paper/
