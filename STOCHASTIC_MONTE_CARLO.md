# Stochastic environment and Monte Carlo

Install the one added dependency from the project root:

```powershell
python -m pip install -r requirements.txt
```

Run one seeded stochastic path with `python run.py`, or run independent seeds
with `python -m experiments.monte_carlo`. The latter writes one row per run to
`results/raw/monte_carlo/` and its configuration and survival estimate to
`results/metadata/monte_carlo/`.

The configured timestep is one month (`stochastic.dt = 1.0`). The environment
uses full-truncation Euler for the CIR yield process, exact GBM steps for unit
storage cost and token price, Beta draws for settlement slippage, Poisson
arrivals for new datasets, and Bernoulli permanent miner failures. The three
diffusion shocks are correlated using the configured 3x3 correlation matrix.
Each simulation seed controls both the ABM decisions and its separate
environment random stream, so identical seed and configuration reproduce a
path.

Storage price and miner reward remain USD-denominated in this version.
`storage_reward_tokens = storage_reward_usd / token_price_usd` is reported to
show token issuance; consequently a token crash changes token units issued but
does not by itself reduce the USD reward or miner profit. Slippage increases
the Endowment's USD settlement debit to `storage_reward / (1 - slippage)`.
This makes the numeraire assumption explicit and avoids double-counting token
price exposure.

The GBM arithmetic drift for storage cost is configured as `log(0.97)` per
month, preserving the prior 3% monthly decline in expected cost. Stochastic
realizations can still rise in a particular month. `arrival_rate` is the mean
number of new datasets per month; arrivals that cannot get all target replicas
are counted as unmet replica demand and remain in the model for repair.

These are proposed research-model choices, not calibrated empirical
parameters. The included 100-run Monte Carlo default is for development and
does not establish a research conclusion. The `stochastic.enabled` switch can
disable these processes to use the earlier deterministic behavior.
