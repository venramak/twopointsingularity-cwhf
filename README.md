# Signal ≠ Value Lab · CWHF v2.0

The companion lab to the whitepaper **Signal ≠ Value: Coherence-Weighted Human Feedback (CWHF), v2.0** (Venkat Rao, 2026).

- **Live lab:** https://cwhf.vercel.app/
- **On the website:** https://twopointsingularity.com/research/cwhf/
- **Whitepaper (PDF):** [`whitepaper/CWHF-Whitepaper-v2.0.pdf`](whitepaper/CWHF-Whitepaper-v2.0.pdf)
- **Earlier pre-registration:** [OSF W7NYH](https://osf.io/w7nyh)

## The question

When people judge under pressure, do their choices drift away from what they themselves choose on reflection, in a consistent direction rather than at random? If they do, collecting more human feedback entrenches the drift instead of averaging it away. The lab separates what is established, what is hypothesised, what is only simulated and what is proposed.

## What is here

| Path | Contents |
| --- | --- |
| `index.html` | The lab: evidence base (71 cases), the Reach room on every digital signal, a flip-test demonstration, the design simulation, the CWHF v2 proposal, method and the claims ledger. One self-contained static page; it stores nothing. |
| `whitepaper/` | Whitepaper v2.0 (PDF, A4). |
| `simulation/` | Code and result files behind every simulated number in the lab and the paper. |
| `archive/` | The retired CWHF Coherence Lab v0.4, kept for the record. Its claims are superseded; see the lab's Claims room. |

## Reproduce the simulation

Python 3.10+ with numpy and scipy:

```
cd simulation
python3 run_h1_grid.py . h1_results.json
python3 persontest.py . h1_person_test.json
python3 h2sim.py 80 h2_results.json
python3 power_sd.py
```

Every simulated result describes the study design under stated assumptions. None of it is evidence about how real people behave.

## Embedding

The lab may be embedded in an iframe on twopointsingularity.com (allowed by `vercel.json`):

```html
<iframe src="https://cwhf.vercel.app/" title="Signal ≠ Value Lab" loading="lazy" style="width:100%;height:85vh;border:0"></iframe>
```

## Licence and citation

Content and code: CC BY 4.0 (see `LICENSE`). Cite with `CITATION.cff`, or:

Rao, V. (2026). *Signal ≠ Value: Coherence-Weighted Human Feedback (CWHF).* Whitepaper v2.0 and companion lab. Two Point Singularity.
