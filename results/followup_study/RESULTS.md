# evaluation experiment results

Scores are fractions. Evaluation streams are averaged within training seeds before mean/sample SD. Shared controls are not independent training replicates. Partial suites are marked by their counts.

| Group | Dataset | Method | Horizon | Metric | Rows | Mean | SD |
|---|---|---|---:|---|---:|---:|---:|
| ablations | longmemeval | difference_prefix |  | token_f1_diagnostic | 3 | 0.2434 | 0.03150557556908401 |
| ablations | longmemeval | difference_prefix |  | correct | 3 | 0.1615 | 0.01339491949631491 |
| ordering | longmemeval | chalm_score_order |  | token_f1_diagnostic | 1 | 0.2493 |  |
| ordering | longmemeval | chalm_score_order |  | correct | 1 | 0.1667 |  |
| ablations | synthetic | chalm | 100 | correct | 3 | 0.7278 | 0.10447399109259115 |
| ablations | synthetic | chalm | 500 | correct | 3 | 0.7289 | 0.08855492820076122 |
| ablations | synthetic | chalm | 1000 | correct | 3 | 0.7444 | 0.0574778540283447 |
| ablations | synthetic | chalm | 5000 | correct | 3 | 0.7244 | 0.06710963942462887 |
| newest-wins-cue_free | synthetic | lexical_only | 1000 | correct | 1 | 0.5667 |  |
| newest-wins-cue_free | synthetic | lexical_only | 5000 | correct | 1 | 0.6067 |  |
| ablations | longmemeval | chalm |  | token_f1_diagnostic | 3 | 0.2551 | 0.015669168352620607 |
| ablations | longmemeval | chalm |  | correct | 3 | 0.1719 | 0.004625924443258076 |
| cue-free | synthetic | chalm | 1000 | correct | 3 | 0.6156 | 0.06194382162118373 |
| cue-free | synthetic | chalm | 5000 | correct | 3 | 0.5933 | 0.029059326290271154 |
| newest-wins-cue_free | synthetic | chalm | 1000 | correct | 3 | 0.6644 | 0.0800925390708449 |
| newest-wins-cue_free | synthetic | chalm | 5000 | correct | 3 | 0.7022 | 0.048227854253801584 |
| ablations | locomo | lexical_only |  | token_f1_diagnostic | 1 | 0.1564 |  |
| ablations | locomo | lexical_only |  | correct | 1 | 0.0559 |  |
| ablations | locomo | lexical_only |  | locomo_official_qa_score | 1 | 0.3100 |  |
| newest-wins-original | locomo | lexical_only |  | token_f1_diagnostic | 1 | 0.1563 |  |
| newest-wins-original | locomo | lexical_only |  | correct | 1 | 0.0564 |  |
| newest-wins-original | locomo | lexical_only |  | locomo_official_qa_score | 1 | 0.3086 |  |
| ordering | locomo | chalm_score_order |  | token_f1_diagnostic | 1 | 0.2380 |  |
| ordering | locomo | chalm_score_order |  | correct | 1 | 0.1098 |  |
| ordering | locomo | chalm_score_order |  | locomo_official_qa_score | 1 | 0.3766 |  |
| ablations | longmemeval | lexical_only |  | token_f1_diagnostic | 1 | 0.1915 |  |
| ablations | longmemeval | lexical_only |  | correct | 1 | 0.1267 |  |
| retrieval | synthetic | bm25_cache | 100 | correct | 1 | 0.7700 |  |
| retrieval | synthetic | bm25_cache | 500 | correct | 1 | 0.7867 |  |
| retrieval | synthetic | bm25_cache | 1000 | correct | 1 | 0.7600 |  |
| retrieval | synthetic | bm25_cache | 5000 | correct | 1 | 0.7533 |  |
| newest-wins-original | synthetic | lexical_only | 100 | correct | 1 | 0.7233 |  |
| newest-wins-original | synthetic | lexical_only | 500 | correct | 1 | 0.6800 |  |
| newest-wins-original | synthetic | lexical_only | 1000 | correct | 1 | 0.6667 |  |
| newest-wins-original | synthetic | lexical_only | 5000 | correct | 1 | 0.6867 |  |
| ablations | locomo | chalm |  | token_f1_diagnostic | 3 | 0.2343 | 0.010730210881399463 |
| ablations | locomo | chalm |  | correct | 3 | 0.1020 | 0.009316340440669969 |
| ablations | locomo | chalm |  | locomo_official_qa_score | 3 | 0.3475 | 0.01816962547735739 |
| ablations | synthetic | lexical_only | 100 | correct | 1 | 0.7233 |  |
| ablations | synthetic | lexical_only | 500 | correct | 1 | 0.6800 |  |
| ablations | synthetic | lexical_only | 1000 | correct | 1 | 0.6533 |  |
| ablations | synthetic | lexical_only | 5000 | correct | 1 | 0.7067 |  |
| newest-wins-original | synthetic | chalm | 100 | correct | 3 | 0.7278 | 0.10447399109259115 |
| newest-wins-original | synthetic | chalm | 500 | correct | 3 | 0.7289 | 0.08855492820076122 |
| newest-wins-original | synthetic | chalm | 1000 | correct | 3 | 0.7289 | 0.07343427396381469 |
| newest-wins-original | synthetic | chalm | 5000 | correct | 3 | 0.7111 | 0.060491811505849806 |
| priority | locomo | hybrid_fifo_rag |  | token_f1_diagnostic | 1 | 0.1966 |  |
| priority | locomo | hybrid_fifo_rag |  | correct | 1 | 0.0760 |  |
| priority | locomo | hybrid_fifo_rag |  | locomo_official_qa_score | 1 | 0.3218 |  |
| ablations | locomo | centered_zero |  | token_f1_diagnostic | 3 | 0.2159 | 0.0 |
| ablations | locomo | centered_zero |  | correct | 3 | 0.0796 | 0.0 |
| ablations | locomo | centered_zero |  | locomo_official_qa_score | 3 | 0.3238 | 0.0 |
| ablations | synthetic | centered_zero | 100 | correct | 3 | 0.7133 | 0.0 |
| ablations | synthetic | centered_zero | 500 | correct | 3 | 0.6978 | 0.0 |
| ablations | synthetic | centered_zero | 1000 | correct | 3 | 0.6400 | 0.0 |
| ablations | synthetic | centered_zero | 5000 | correct | 3 | 0.6667 | 0.0 |
| ablations | longmemeval | uncentered |  | token_f1_diagnostic | 3 | 0.2452 | 0.031252550332276474 |
| ablations | longmemeval | uncentered |  | correct | 3 | 0.1630 | 0.015765775299105753 |
| ablations | locomo | uncentered |  | token_f1_diagnostic | 3 | 0.2216 | 0.02562024381412462 |
| ablations | locomo | uncentered |  | correct | 3 | 0.0925 | 0.014155522534559078 |
| ablations | locomo | uncentered |  | locomo_official_qa_score | 3 | 0.3407 | 0.020571318038192128 |
| ablations | synthetic | difference_prefix | 100 | correct | 3 | 0.7289 | 0.06049181150584978 |
| ablations | synthetic | difference_prefix | 500 | correct | 3 | 0.7481 | 0.10138140510178145 |
| ablations | synthetic | difference_prefix | 1000 | correct | 3 | 0.5844 | 0.34165176119248375 |
| ablations | synthetic | difference_prefix | 5000 | correct | 3 | 0.6844 | 0.18503252967258768 |
| newest-wins-original | locomo | chalm |  | token_f1_diagnostic | 3 | 0.2389 | 0.011750926425739876 |
| newest-wins-original | locomo | chalm |  | correct | 3 | 0.1047 | 0.010513903835760877 |
| newest-wins-original | locomo | chalm |  | locomo_official_qa_score | 3 | 0.3511 | 0.01944426273607028 |
| ordering | synthetic | chalm_score_order | 100 | correct | 1 | 0.8467 |  |
| ordering | synthetic | chalm_score_order | 500 | correct | 1 | 0.8267 |  |
| ordering | synthetic | chalm_score_order | 1000 | correct | 1 | 0.8267 |  |
| ordering | synthetic | chalm_score_order | 5000 | correct | 1 | 0.7733 |  |
| ablations | locomo | difference_prefix |  | token_f1_diagnostic | 3 | 0.2215 | 0.02464962918629434 |
| ablations | locomo | difference_prefix |  | correct | 3 | 0.0911 | 0.015130895469603944 |
| ablations | locomo | difference_prefix |  | locomo_official_qa_score | 3 | 0.3396 | 0.01922908182004587 |
| priority | longmemeval | hybrid_growing_rag |  | token_f1_diagnostic | 1 | 0.2545 |  |
| priority | longmemeval | hybrid_growing_rag |  | correct | 1 | 0.1822 |  |
| retrieval | longmemeval | bm25_cache |  | token_f1_diagnostic | 1 | 0.1998 |  |
| retrieval | longmemeval | bm25_cache |  | correct | 1 | 0.1356 |  |
| ablations | locomo | hybrid_text |  | token_f1_diagnostic | 3 | 0.2021 | 0.0 |
| ablations | locomo | hybrid_text |  | correct | 3 | 0.0796 | 0.0 |
| ablations | locomo | hybrid_text |  | locomo_official_qa_score | 3 | 0.3374 | 0.0 |
| priority | longmemeval | hybrid_fifo_rag |  | token_f1_diagnostic | 1 | 0.2324 |  |
| priority | longmemeval | hybrid_fifo_rag |  | correct | 1 | 0.1578 |  |
| priority | locomo | hybrid_growing_rag |  | token_f1_diagnostic | 1 | 0.2162 |  |
| priority | locomo | hybrid_growing_rag |  | correct | 1 | 0.0811 |  |
| priority | locomo | hybrid_growing_rag |  | locomo_official_qa_score | 1 | 0.3371 |  |
| newest-wins-original | longmemeval | chalm |  | token_f1_diagnostic | 3 | 0.2548 | 0.014893235745943374 |
| newest-wins-original | longmemeval | chalm |  | correct | 3 | 0.1711 | 0.004444444444444445 |
| retrieval | locomo | bm25_growing_rag |  | token_f1_diagnostic | 1 | 0.2026 |  |
| retrieval | locomo | bm25_growing_rag |  | correct | 1 | 0.0846 |  |
| retrieval | locomo | bm25_growing_rag |  | locomo_official_qa_score | 1 | 0.3241 |  |
| ablations | longmemeval | hybrid_text |  | token_f1_diagnostic | 3 | 0.2180 | 0.0 |
| ablations | longmemeval | hybrid_text |  | correct | 3 | 0.1511 | 0.0 |
| retrieval | locomo | bm25_cache |  | token_f1_diagnostic | 1 | 0.1860 |  |
| retrieval | locomo | bm25_cache |  | correct | 1 | 0.0745 |  |
| retrieval | locomo | bm25_cache |  | locomo_official_qa_score | 1 | 0.3214 |  |
| retrieval | synthetic | bm25_growing_rag | 100 | correct | 1 | 0.6867 |  |
| retrieval | synthetic | bm25_growing_rag | 500 | correct | 1 | 0.7022 |  |
| retrieval | synthetic | bm25_growing_rag | 1000 | correct | 1 | 0.6600 |  |
| retrieval | synthetic | bm25_growing_rag | 5000 | correct | 1 | 0.6933 |  |
| ablations | synthetic | uncentered | 100 | correct | 3 | 0.7167 | 0.05174724898753344 |
| ablations | synthetic | uncentered | 500 | correct | 3 | 0.7437 | 0.09700027576219765 |
| ablations | synthetic | uncentered | 1000 | correct | 3 | 0.5622 | 0.3842645693519756 |
| ablations | synthetic | uncentered | 5000 | correct | 3 | 0.5422 | 0.4183211596280932 |
| ablations | synthetic | hybrid_text | 100 | correct | 3 | 0.7833 | 0.0 |
| ablations | synthetic | hybrid_text | 500 | correct | 3 | 0.7822 | 0.0 |
| ablations | synthetic | hybrid_text | 1000 | correct | 3 | 0.7267 | 0.0 |
| ablations | synthetic | hybrid_text | 5000 | correct | 3 | 0.7533 | 0.0 |
| ablations | longmemeval | centered_zero |  | token_f1_diagnostic | 3 | 0.2096 | 0.0 |
| ablations | longmemeval | centered_zero |  | correct | 3 | 0.1422 | 0.0 |
| priority | synthetic | hybrid_growing_rag | 100 | correct | 1 | 0.6467 |  |
| priority | synthetic | hybrid_growing_rag | 500 | correct | 1 | 0.6889 |  |
| priority | synthetic | hybrid_growing_rag | 1000 | correct | 1 | 0.6400 |  |
| priority | synthetic | hybrid_growing_rag | 5000 | correct | 1 | 0.6200 |  |
| priority | synthetic | hybrid_fifo_rag | 100 | correct | 1 | 0.6467 |  |
| priority | synthetic | hybrid_fifo_rag | 500 | correct | 1 | 0.6889 |  |
| priority | synthetic | hybrid_fifo_rag | 1000 | correct | 1 | 0.1200 |  |
| priority | synthetic | hybrid_fifo_rag | 5000 | correct | 1 | 0.0000 |  |
| newest-wins-original | longmemeval | lexical_only |  | token_f1_diagnostic | 1 | 0.1915 |  |
| newest-wins-original | longmemeval | lexical_only |  | correct | 1 | 0.1267 |  |
| retrieval | longmemeval | bm25_growing_rag |  | token_f1_diagnostic | 1 | 0.2261 |  |
| retrieval | longmemeval | bm25_growing_rag |  | correct | 1 | 0.1556 |  |
