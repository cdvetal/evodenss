# Statistical analysis for the Scientific Reports article

Scripts that reproduce the paired comparison between self-supervised and supervised
neuroevolution reported in *Assessing the role of label availability in self-supervised
and supervised neuroevolution* (Vinhas, Correia & Machado, Scientific Reports).

Input: `srep_paper_results_final_checkpoints.xlsx`, the per-run results file released
with the article (same Zenodo record). Runs are paired by run id; run *k* of every SSL
setup and of the supervised baseline uses seed *k* (see the `README` sheet of that file).

| Script | Produces |
|---|---|
| `paired_analysis.py` | Table 5 and Figure 4 statistics: per-run final accuracies, mean paired differences, 95% CIs, two-sided paired t-tests and Holm-corrected p-values (eight setups per scenario). |
| `fig4.py` | Figure 4 of the main text (reads the per-run accuracies and plots the paired differences). |
| `all_setups_by_budget.py` | Supplementary Appendix D: the same test repeated at every epoch-matched budget (SSL p pretext + d downstream vs supervised p + d epochs). |
| `budget_trellis.py` | Supplementary Figure 5 (Appendix D) from the output of `all_setups_by_budget.py`. |

```bash
pip install pandas numpy scipy matplotlib openpyxl
python paired_analysis.py srep_paper_results_final_checkpoints.xlsx --rule final --out results_final
python fig4.py srep_paper_results_final_checkpoints.xlsx figure4 final 500 600 1100
python all_setups_by_budget.py srep_paper_results_final_checkpoints.xlsx all_setups_by_budget
python budget_trellis.py all_setups_by_budget.csv budget_trellis
```

`--rule peak` reproduces the superseded first-revision analysis (per-run best checkpoint) for traceability only.
