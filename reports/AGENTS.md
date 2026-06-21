<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-06-21 | Updated: 2026-06-21 -->

# reports

## Purpose
Generated output: written analysis reports (Markdown, some with Korean `_ko`
variants) and machine-readable result tables (CSV / TXT). One report typically
corresponds to one root analysis script. This directory is a sink — files here
are produced by scripts, not authored by hand.

## Contents Overview
| Pattern | Description |
|---------|-------------|
| `*_report.md` / `*_report_ko.md` | Narrative findings per experiment (English / Korean) |
| `2026-05-*.md`, `2026-05-27_*.md` | Dated analysis writeups (AAPL, assumption, intervention, LW10, timeseries, K-1) |
| `intervention_ols_*.txt` | statsmodels OLS summaries per estimator (Sample/LW/Gerber) |
| `*_table.csv`, `*_results.csv`, `*_summary.csv` | Tabular results (coefficients, snapshots, robustness) |
| `crisis_weight_test_topassets.txt`, `*_topassets.txt` | Ranked asset lists from tests |
| `integrated_report_ko.md` | Combined cross-experiment summary |

## For AI Agents

### Working In This Directory
- Treat as regenerable output. To change a report, edit the producing script and re-run it — do not hand-edit then re-run (the run will overwrite).
- Filenames encode provenance (experiment name, sometimes date). Match that convention when a script writes a new report.

## Dependencies

### Internal
- Produced by the root analysis scripts (see `../AGENTS.md`).

<!-- MANUAL: -->
