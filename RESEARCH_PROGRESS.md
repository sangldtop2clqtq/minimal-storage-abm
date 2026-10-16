# Research Progress Report

**Project:** Sustainable Perpetual Storage Minimal ABM  
**Report date:** 2026-10-15  
**Scope:** Progress based on the current source files and the work completed so far. “Done” means implemented in the current code, not empirically validated.

| Task | Expected output | Priority | Status | Research |
|---|---|---:|---|---|
| Define the initial ABM scope | Minimal model covering miners, datasets, pricing, exits, repair, Endowment, and metrics | High | Done | Keeps the first implementation focused on the economic mechanism rather than blockchain/network protocol details. |
| Implement `MinerAgent` | Miner state, capacity use, reward, operating cost, profit, leave probability, and active/offline status | High | Partial | The basic economics and exit behavior exist. Leave behavior still needs verification against its intended probability and model-level sampling. |
| Implement `Dataset` | Dataset size, target replicas, current replica count, deficit, and healthy/degraded/lost state | High | Done | Supports measuring replication compliance and availability. |
| Implement bonding-curve pricing | Storage price responds to utilization and is bounded by configured minimum and maximum | High | Done | Implements the proposed sigmoid pricing mechanism; parameters are currently synthetic and uncalibrated. |
| Implement miner exit behavior | Profit-dependent probabilistic miner exit and removal of hosted replicas | High | Implemented; needs validation | Model compares a seeded uniform draw in [0, 1) with the leave probability. Forced exits are also supported for stress scenarios. |
| Implement dynamic repair mechanism | Deficits trigger repair rewards and eligible miners probabilistically restore replicas | High | Partial | Repair code exists, but current config uses zero reward, zero alpha, and zero beta. With beta zero, repair participation is a 50% coin flip; this is not yet a meaningful calibrated incentive experiment. |
| Implement Endowment accounting | Endowment earns fixed yield and pays storage and repair rewards each month | High | Partial | Monthly accounting is implemented. Funding assumptions and whether payouts match a sustainable budget need further validation. |
| Implement monthly operating-cost decline | Miner unit cost follows `C_(t+1) = C_t × (1 - decline_rate)` | Medium | Done | Current assumption is a 3% monthly reduction. This is a proposed scenario parameter inspired by a Moore-like decline, not a direct empirical implication of Moore’s law for storage costs. |
| Add reproducible random seed | Same seed and configuration reproduce the same random draws | High | Done | The model uses a local `random.Random` instance and currently configures seed 46. Reproducibility should still be confirmed with a dedicated check. |
| Add monthly metrics | Track active miners, capacity, utilization, price, replica health, availability, repair, costs, and Endowment | High | Done | Metrics are stored in `model.metrics`; experiment scripts now export CSV and metadata. Figures are not generated yet. |
| Add a runnable entry point | `python run.py` runs the configured horizon and prints a summary | High | Done | The current run script reports months, seed, final active miners, availability, and Endowment. |
| Run baseline scenario | 100 miners, 10 datasets, 12 months, no forced shock | High | Script implemented; run pending | `python -m experiments.baseline` runs the configured case and writes monthly CSV plus JSON metadata. Results need inspection and calibration. |
| Compare fixed and deficit-sensitive repair rewards | Recovery and availability comparison under identical conditions | High | Not started | This is the first research hypothesis identified in the project guide; it requires meaningful repair parameters and repeated seeds. |
| Run forced miner-exit stress test | Trigger a specified miner exit shock and measure recovery | High | Script implemented; run pending | `python -m experiments.miner_exit` forces 30% of active miners offline at model month 6 and records forced exits in metrics. |
| Add automated tests | Tests for formulas, invariants, capacity, repair, and reproducibility | High | Tests implemented; execution pending | Tests now cover mechanisms, agent behavior, dataset states, model invariants, forced exits, and seeded reproducibility. Run with `python -m pytest`. |
| Export and visualize results | CSV output and time-series plots for miners, utilization, replicas, repair, price, and Endowment | Medium | CSV/metadata implemented; figures pending | Experiment helper writes monthly raw CSV and JSON metadata. Figures and summary tables remain to be added. |
| Add miner joining | New miner agents enter when expected profitability justifies joining | Medium | Not started | Without joining, active capacity can only stay constant or decline, limiting equilibrium and recovery analysis. |
| Add UserAgent and endogenous demand | Users submit datasets and generate storage/retrieval demand | Medium | Not started | Current datasets are generated at initialization; user behavior and demand response are exogenous or absent. |
| Add stochastic environment and token settlement | Stochastic yield/cost/token price/slippage paths | Low (later phase) | Not started | The guide places CIR/GBM and token settlement after a verified deterministic baseline. |
| Add Risk Buffer, Insurance, slashing, and retrieval rewards | Extended risk and incentive mechanisms | Low (later phase) | Not started | These extensions should follow core ABM validation and be introduced one at a time for attribution. |
| Add Monte Carlo and long-horizon analysis | Multiple seeds and long-run sustainability estimates | Low (later phase) | Not started | Required for robustness; a single seeded run cannot establish sustainability or mechanism effectiveness. |

## Current assessment

The project now has a runnable first model, baseline and forced-exit experiment scripts, result/metadata output, and an initial pytest suite. It remains a prototype: tests and experiments have not yet been run as a suite, repair parameters are currently degenerate, experiment comparisons are missing, and sustainability has not been established.

## Recommended next steps

1. Run `python -m pytest` and fix any failing tests.
2. Choose and justify non-degenerate repair reward and sensitivity parameters.
3. Run the baseline and forced-exit experiments; inspect CSV and metadata outputs.
4. Add fixed-versus-dynamic repair comparison across paired seeds.
5. Add plots and sensitivity analysis before drawing research conclusions.
