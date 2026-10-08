# Paired significance results

Differences and CIs are fractions (multiply by 100 for percentage points). CIs are pointwise; Holm correction applies only to aggregate comparisons; per-seed adjusted p is blank. Aggregate inference is conditional on the fitted checkpoints, not a population claim over training seeds. LoCoMo has only ten conversation clusters.

| Dataset | Horizon | Comparison | Seed | Delta | 95% CI | Raw p | Holm p |
|---|---:|---|---|---:|---|---:|---:|
| locomo |  | chalm vs rag | 13 | 0.0541 | [0.0381, 0.0706] | 0.0025 |  |
| locomo |  | chalm vs rag | 29 | 0.0224 | [0.0069, 0.0379] | 0.026 |  |
| locomo |  | chalm vs rag | 31 | 0.0227 | [0.0069, 0.0429] | 0.0215 |  |
| locomo |  | chalm vs rag | mean | 0.0331 | [0.0187, 0.0490] | 0.0049 | 0.025197480251974803 |
| longmemeval |  | chalm vs rag | 13 | 0.0669 | [0.0298, 0.1052] | 0.0003 |  |
| longmemeval |  | chalm vs rag | 29 | 0.0365 | [0.0038, 0.0701] | 0.031 |  |
| longmemeval |  | chalm vs rag | 31 | 0.0581 | [0.0259, 0.0908] | 0.0003 |  |
| longmemeval |  | chalm vs rag | mean | 0.0538 | [0.0226, 0.0855] | 0.0005 | 0.003999600039996 |
| synthetic | 100 | chalm vs rag | 13 | 0.1600 | [0.1200, 0.2033] | 9.999e-05 |  |
| synthetic | 100 | chalm vs rag | 29 | 0.0300 | [-0.0267, 0.0900] | 0.3662 |  |
| synthetic | 100 | chalm vs rag | 31 | -0.0467 | [-0.1033, 0.0100] | 0.1297 |  |
| synthetic | 100 | chalm vs rag | mean | 0.0478 | [0.0033, 0.0944] | 0.0432 | 0.1295870412958704 |
| synthetic | 1000 | chalm vs rag | 13 | 0.1867 | [0.1267, 0.2533] | 9.999e-05 |  |
| synthetic | 1000 | chalm vs rag | 29 | 0.1133 | [0.0465, 0.1867] | 0.0041 |  |
| synthetic | 1000 | chalm vs rag | 31 | 0.0733 | [0.0067, 0.1400] | 0.05599 |  |
| synthetic | 1000 | chalm vs rag | mean | 0.1244 | [0.0667, 0.1844] | 9.999e-05 | 0.0011998800119988001 |
| synthetic | 500 | chalm vs rag | 13 | 0.1956 | [0.1467, 0.2489] | 9.999e-05 |  |
| synthetic | 500 | chalm vs rag | 29 | 0.0444 | [-0.0267, 0.1156] | 0.2681 |  |
| synthetic | 500 | chalm vs rag | 31 | 0.0400 | [-0.0222, 0.1022] | 0.2622 |  |
| synthetic | 500 | chalm vs rag | mean | 0.0933 | [0.0415, 0.1452] | 0.0012 | 0.008399160083991601 |
| synthetic | 5000 | chalm vs rag | 13 | 0.0733 | [0.0067, 0.1467] | 0.06519 |  |
| synthetic | 5000 | chalm vs rag | 29 | 0.2067 | [0.1400, 0.2733] | 9.999e-05 |  |
| synthetic | 5000 | chalm vs rag | 31 | 0.1533 | [0.0867, 0.2267] | 9.999e-05 |  |
| synthetic | 5000 | chalm vs rag | mean | 0.1444 | [0.0889, 0.2044] | 9.999e-05 | 0.0011998800119988001 |
| locomo |  | chalm vs linearrag_local | 13 | 0.0483 | [0.0237, 0.0726] | 0.0109 |  |
| locomo |  | chalm vs linearrag_local | 29 | 0.0166 | [-0.0059, 0.0378] | 0.217 |  |
| locomo |  | chalm vs linearrag_local | 31 | 0.0169 | [-0.0116, 0.0451] | 0.2973 |  |
| locomo |  | chalm vs linearrag_local | mean | 0.0273 | [0.0029, 0.0510] | 0.08249 | 0.1341865813418658 |
| longmemeval |  | chalm vs linearrag_local | 13 | 0.0899 | [0.0533, 0.1262] | 9.999e-05 |  |
| longmemeval |  | chalm vs linearrag_local | 29 | 0.0594 | [0.0255, 0.0941] | 0.0013 |  |
| longmemeval |  | chalm vs linearrag_local | 31 | 0.0810 | [0.0472, 0.1153] | 9.999e-05 |  |
| longmemeval |  | chalm vs linearrag_local | mean | 0.0768 | [0.0456, 0.1088] | 9.999e-05 | 0.0011998800119988001 |
| synthetic | 100 | chalm vs linearrag_local | 13 | 0.1667 | [0.1267, 0.2100] | 9.999e-05 |  |
| synthetic | 100 | chalm vs linearrag_local | 29 | 0.0367 | [-0.0200, 0.0967] | 0.2677 |  |
| synthetic | 100 | chalm vs linearrag_local | 31 | -0.0400 | [-0.0967, 0.0200] | 0.2181 |  |
| synthetic | 100 | chalm vs linearrag_local | mean | 0.0544 | [0.0089, 0.1022] | 0.0243 | 0.09719028097190281 |
| synthetic | 1000 | chalm vs linearrag_local | 13 | 0.1933 | [0.1267, 0.2600] | 9.999e-05 |  |
| synthetic | 1000 | chalm vs linearrag_local | 29 | 0.1200 | [0.0533, 0.1933] | 0.0021 |  |
| synthetic | 1000 | chalm vs linearrag_local | 31 | 0.0800 | [0.0133, 0.1533] | 0.0383 |  |
| synthetic | 1000 | chalm vs linearrag_local | mean | 0.1311 | [0.0733, 0.1956] | 9.999e-05 | 0.0011998800119988001 |
| synthetic | 500 | chalm vs linearrag_local | 13 | 0.1511 | [0.1067, 0.2000] | 9.999e-05 |  |
| synthetic | 500 | chalm vs linearrag_local | 29 | 0.0000 | [-0.0711, 0.0667] | 1 |  |
| synthetic | 500 | chalm vs linearrag_local | 31 | -0.0044 | [-0.0667, 0.0578] | 1 |  |
| synthetic | 500 | chalm vs linearrag_local | mean | 0.0489 | [-0.0000, 0.1007] | 0.06709 | 0.1341865813418658 |
| synthetic | 5000 | chalm vs linearrag_local | 13 | 0.0067 | [-0.0600, 0.0733] | 1 |  |
| synthetic | 5000 | chalm vs linearrag_local | 29 | 0.1400 | [0.0867, 0.2000] | 9.999e-05 |  |
| synthetic | 5000 | chalm vs linearrag_local | 31 | 0.0867 | [0.0267, 0.1533] | 0.0147 |  |
| synthetic | 5000 | chalm vs linearrag_local | mean | 0.0778 | [0.0289, 0.1311] | 0.0042 | 0.025197480251974803 |
