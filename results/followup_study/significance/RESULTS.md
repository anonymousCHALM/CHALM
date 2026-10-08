# Follow-up paired significance

Differences/CI are fractions. Holm applies only to aggregate comparisons; per-seed adjusted p is blank. Aggregates are conditional on fitted checkpoints. LongMemEval uses diagnostic token F1; LoCoMo uses QA score.

| Dataset | Horizon | Comparison | Seed | Checkpoints | Complete | Difference | 95% CI | Raw p | Holm p |
|---|---|---|---|---|---|---|---|---|---|
| synthetic | 5000 | chalm vs hybrid_growing_rag | 13 | 13 | True | 0.03333 | [-0.05333, 0.12000] | 0.54975 |  |
| synthetic | 5000 | chalm vs hybrid_growing_rag | 29 | 29 | True | 0.16667 | [0.10000, 0.23333] | 9.999e-05 |  |
| synthetic | 5000 | chalm vs hybrid_growing_rag | 31 | 31 | True | 0.11333 | [0.04667, 0.18000] | 0.0023998 |  |
| synthetic | 5000 | chalm vs hybrid_growing_rag | mean | 13,29,31 | True | 0.10444 | [0.04000, 0.16889] | 0.0019998 | 0.007999200079992 |
| longmemeval |  | chalm vs hybrid_growing_rag | 13 | 13 | True | 0.01376 | [-0.01591, 0.04426] | 0.37556 |  |
| longmemeval |  | chalm vs hybrid_growing_rag | 29 | 29 | True | -0.01670 | [-0.04372, 0.01071] | 0.23338 |  |
| longmemeval |  | chalm vs hybrid_growing_rag | 31 | 31 | True | 0.00492 | [-0.02155, 0.03181] | 0.71503 |  |
| longmemeval |  | chalm vs hybrid_growing_rag | mean | 13,29,31 | True | 0.00066 | [-0.02290, 0.02456] | 0.954 | 0.954004599540046 |
| locomo |  | chalm vs hybrid_growing_rag | 13 | 13 | True | 0.03131 | [0.01808, 0.04144] | 0.0067993 |  |
| locomo |  | chalm vs hybrid_growing_rag | 29 | 29 | True | -0.00032 | [-0.01620, 0.01355] | 0.9759 |  |
| locomo |  | chalm vs hybrid_growing_rag | 31 | 31 | True | -0.00000 | [-0.01690, 0.01581] | 1 |  |
| locomo |  | chalm vs hybrid_growing_rag | mean | 13,29,31 | True | 0.01033 | [-0.00324, 0.02121] | 0.18328 | 0.3665633436656334 |
| longmemeval |  | chalm vs bm25_cache | 13 | 13 | True | 0.06847 | [0.03836, 0.09872] | 9.999e-05 |  |
| longmemeval |  | chalm vs bm25_cache | 29 | 29 | True | 0.03801 | [0.01062, 0.06500] | 0.0058994 |  |
| longmemeval |  | chalm vs bm25_cache | 31 | 31 | True | 0.05963 | [0.03325, 0.08564] | 9.999e-05 |  |
| longmemeval |  | chalm vs bm25_cache | mean | 13,29,31 | True | 0.05537 | [0.03141, 0.07894] | 9.999e-05 | 0.0004999500049995 |
| locomo |  | chalm vs bm25_cache | 13 | 13 | True | 0.04704 | [0.03330, 0.06070] | 0.0024998 |  |
| locomo |  | chalm vs bm25_cache | 29 | 29 | True | 0.01541 | [0.00384, 0.02668] | 0.047095 |  |
| locomo |  | chalm vs bm25_cache | 31 | 31 | True | 0.01573 | [-0.00614, 0.03673] | 0.18828 |  |
| locomo |  | chalm vs bm25_cache | mean | 13,29,31 | True | 0.02606 | [0.01205, 0.04041] | 0.0063994 | 0.019198080191980802 |
