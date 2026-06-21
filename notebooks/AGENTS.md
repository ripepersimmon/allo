<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-06-21 | Updated: 2026-06-21 -->

# notebooks

## Purpose
Exploratory Jupyter notebooks. The interactive counterpart to the root scripts —
used for visual exploration of estimator behaviour and crisis-period GMV weights
before that logic is hardened into a standalone script.

## Key Files
| File | Description |
|------|-------------|
| `crisis_study.ipynb` | Main exploratory notebook: long-only rolling GMV across estimators and the three crisis windows |

## For AI Agents

### Working In This Directory
- Launch from repo root with the venv active: `jupyter notebook notebooks/crisis_study.ipynb`.
- Notebooks insert the repo root on `sys.path` and import from `src/`; run cells from the repo root so relative `sp500/` and `results/` paths resolve.
- The notebook uses long-only (`constrained=True`) rolling GMV as the default, unlike `beta_weight.py` which uses the unconstrained solver.
- Keep durable logic in `src/`; the notebook should orchestrate and visualize, not redefine estimators or solvers.

### Common Patterns
- `.ipynb_checkpoints/` is editor scratch — ignore and do not document it.

## Dependencies

### Internal
- `src/` (data_loader, estimators, portfolio, analysis)

### External
- jupyter, matplotlib, seaborn

<!-- MANUAL: -->
