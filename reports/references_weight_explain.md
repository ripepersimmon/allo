# References — LW-GMV Weight-Explanation Study

Prior research relevant to the finding that **long-only LW-GMV portfolio weight is
primarily a low-beta tilt** (see `2026-06-21_weight_explain_results.md`). All entries
verified against publisher / SSRN metadata (June 2026).

---

## Primary anchor — minimum-variance portfolio composition

**[CST2011] Clarke, R., de Silva, H., & Thorley, S. (2011). "Minimum-Variance
Portfolio Composition." *The Journal of Portfolio Management*, 37(2), 31–45.**
DOI: 10.3905/jpm.2011.37.2.031 · [link](https://jpm.pm-research.com/content/37/2/31)

> **Why it matters (direct match).** Derives the *analytic* long-only minimum-variance
> weight under a single-factor covariance model. Result: optimal weight is a simple
> function of beta — **securities with market beta above an analytically specified
> threshold are driven out of the long-only solution entirely**, and only a small set
> of low-beta names remain with positive weight. This is exactly our empirical finding:
> the SHAP beta-dependence plot shows a hard **threshold/cliff** (positive weight
> contribution for β ≲ 0.7, flat near-zero floor for β ≳ 1), and beta dominates every
> importance metric. CST2011 is the theoretical explanation for our nonlinear result
> (and for why the gradient-boosted M3 doubles the linear R²).

---

## Low-beta / low-volatility anomaly (why a low-beta tilt is economically interesting)

**[FP2014] Frazzini, A., & Pedersen, L. H. (2014). "Betting Against Beta."
*Journal of Financial Economics*, 111(1), 1–25.**
DOI: 10.1016/j.jfineco.2013.10.005 · [link](https://ideas.repec.org/a/eee/jfinec/v111y2014i1p1-25.html)

> Leverage-constrained investors bid up high-beta assets, so high beta earns low alpha;
> a long-low-beta / short-high-beta (BAB) factor earns positive risk-adjusted returns.
> Frames the GMV low-beta tilt as exposure to the documented "defensive equity" premium,
> not just a variance-minimization artifact.

**[BBW2011] Baker, M., Bradley, B., & Wurgler, J. (2011). "Benchmarks as Limits to
Arbitrage: Understanding the Low-Volatility Anomaly." *Financial Analysts Journal*,
67(1), 40–54.**
DOI: 10.2469/faj.v67.n1.4 · [link](https://rpc.cfainstitute.org/research/financial-analysts-journal/2011/benchmarks-as-limits-to-arbitrage-understanding-the-low-volatility-anomaly)

> Low-beta / low-volatility stocks have historically outperformed; fixed-benchmark
> mandates limit arbitrage of the anomaly. Complements FP2014 on why the low-beta tilt
> persists.

---

## Estimator & constraint (why our design choices are standard / why the tautology is mild)

**[LW2004] Ledoit, O., & Wolf, M. (2004). "Honey, I Shrunk the Sample Covariance
Matrix." *The Journal of Portfolio Management*, 30(4), 110–119.**
DOI: 10.3905/jpm.2004.110 · [link](https://jpm.pm-research.com/content/30/4/110)

> The shrinkage covariance estimator used throughout this project (`src/estimators.lw_cov`
> via sklearn `LedoitWolf`). Shrinks extreme sample-covariance coefficients toward a
> structured target, reducing the estimation error that destabilizes mean-variance
> optimizers.

**[JM2003] Jagannathan, R., & Ma, T. (2003). "Risk Reduction in Large Portfolios: Why
Imposing the Wrong Constraints Helps." *The Journal of Finance*, 58(4), 1651–1684.**
DOI: 10.1111/1540-6261.00580 · [link](https://onlinelibrary.wiley.com/doi/10.1111/1540-6261.00580)

> The long-only (no-short) constraint acts as implicit regularization on the covariance.
> **Directly explains our B0 result**: the long-only constraint breaks the clean
> unconstrained identity `w ∝ Σ⁻¹1`, which is why two variance summary statistics
> (`total_var`, `syst_share`) explain only R²≈0.06 of the long-only weight cross-section
> — i.e., the tautology that invalidated the old level-on-variance regressions is *mild*
> for constrained weights.

---

## Crisis correlation dynamics (why portfolio beta rises into the peak)

**[LS2001] Longin, F., & Solnik, B. (2001). "Extreme Correlation of International
Equity Markets." *The Journal of Finance*, 56(2), 649–676.**
DOI: 10.1111/0022-1082.00340 · [link](https://onlinelibrary.wiley.com/doi/abs/10.1111/0022-1082.00340)

> Equity correlation rises in bear markets (not bull markets). **Explains the crisis
> case-study finding** that the LW-GMV portfolio beta *rises* into the most severe peaks
> (COVID, GFC-Lehman) instead of falling: as correlations spike, the cross-section of
> betas compresses toward 1, so even a "low-beta" minimum-variance portfolio carries a
> higher beta at the peak. The de-risking is in *relative* (cross-sectional) terms, not
> absolute beta.

## Optimization vs naive diversification (broader context)

**[DGU2009] DeMiguel, V., Garlappi, L., & Uppal, R. (2009). "Optimal Versus Naive
Diversification: How Inefficient Is the 1/N Portfolio Strategy?" *The Review of
Financial Studies*, 22(5), 1915–1953.**
DOI: 10.1093/rfs/hhm075 · [link](https://academic.oup.com/rfs/article-abstract/22/5/1915/1592901)

> Out-of-sample, estimation error often offsets the gains from optimization vs the 1/N
> rule. Motivates our **out-of-sample (LOYO)** evaluation rather than in-sample fit.

---

## Related (in-repo)

**[KIM2025] Kim et al. (2025). "Estimating Covariance for Global Minimum Variance
Portfolio: A Decision-Focused Learning Approach." arXiv:2508.10776.**
[link](https://arxiv.org/abs/2508.10776) · local: `2508.10776v1.pdf`

> The repo's reference paper; source of the BBC precision-matrix permutation
> (`src/estimators.bbc_permutation`). Tangential to the weight-explanation finding but
> motivates the precision-matrix / GMV framing.

---

## BibTeX

```bibtex
@article{clarke2011minimum,
  title={Minimum-Variance Portfolio Composition},
  author={Clarke, Roger and de Silva, Harindra and Thorley, Steven},
  journal={The Journal of Portfolio Management}, volume={37}, number={2},
  pages={31--45}, year={2011}, doi={10.3905/jpm.2011.37.2.031}}

@article{frazzini2014betting,
  title={Betting Against Beta},
  author={Frazzini, Andrea and Pedersen, Lasse Heje},
  journal={Journal of Financial Economics}, volume={111}, number={1},
  pages={1--25}, year={2014}, doi={10.1016/j.jfineco.2013.10.005}}

@article{baker2011benchmarks,
  title={Benchmarks as Limits to Arbitrage: Understanding the Low-Volatility Anomaly},
  author={Baker, Malcolm and Bradley, Brendan and Wurgler, Jeffrey},
  journal={Financial Analysts Journal}, volume={67}, number={1},
  pages={40--54}, year={2011}, doi={10.2469/faj.v67.n1.4}}

@article{ledoit2004honey,
  title={Honey, I Shrunk the Sample Covariance Matrix},
  author={Ledoit, Olivier and Wolf, Michael},
  journal={The Journal of Portfolio Management}, volume={30}, number={4},
  pages={110--119}, year={2004}, doi={10.3905/jpm.2004.110}}

@article{jagannathan2003risk,
  title={Risk Reduction in Large Portfolios: Why Imposing the Wrong Constraints Helps},
  author={Jagannathan, Ravi and Ma, Tongshu},
  journal={The Journal of Finance}, volume={58}, number={4},
  pages={1651--1684}, year={2003}, doi={10.1111/1540-6261.00580}}

@article{demiguel2009optimal,
  title={Optimal Versus Naive Diversification: How Inefficient Is the 1/N Portfolio Strategy?},
  author={DeMiguel, Victor and Garlappi, Lorenzo and Uppal, Raman},
  journal={The Review of Financial Studies}, volume={22}, number={5},
  pages={1915--1953}, year={2009}, doi={10.1093/rfs/hhm075}}

@article{longin2001extreme,
  title={Extreme Correlation of International Equity Markets},
  author={Longin, Fran\c{c}ois and Solnik, Bruno},
  journal={The Journal of Finance}, volume={56}, number={2},
  pages={649--676}, year={2001}, doi={10.1111/0022-1082.00340}}

@article{kim2025estimating,
  title={Estimating Covariance for Global Minimum Variance Portfolio: A Decision-Focused Learning Approach},
  author={Kim and others}, journal={arXiv preprint arXiv:2508.10776}, year={2025}}
```
