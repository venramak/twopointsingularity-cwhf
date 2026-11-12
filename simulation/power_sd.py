"""Power of the H1 primary test (one-sided one-sample t-test, alpha .05) as a function of the
between-person spread of D_i, for a measured drift of 3 percentage points. Reproduces the
sensitivity table in the whitepaper (Section 7) and the lab's Method room."""
import numpy as np
from scipy import stats
def power(N, d, s, a=.05):
    df = N - 1; nc = d / (s / np.sqrt(N)); tc = stats.t.ppf(1 - a, df)
    return 1 - stats.nct.cdf(tc, df, nc)
for s in (8.2, 8.7, 9.3, 10.0, 10.7, 12.0):
    n80 = next(N for N in range(10, 400) if power(N, 3.0, s) >= .8)
    print(f"SD {s:4.1f}: power at N=60 {power(60, 3.0, s):.2f}; N for 80% = {n80}")
print("SD if sessions were independent (p=.5, ~44 answered items):", round(100*np.sqrt(2*.25/44), 1))
