CONFIG={
    "simulation":{"months":12,"seed":46},
    "miners": {
        "count": 100,
        "capacity_min": 100,
        "capacity_max": 500,
        "cost_min": 0.1,
        "cost_max": 0.2,
        "monthly_cost_decline": 0.03,
    },
    "datasets": {
        "count": 10,
        "size_min": 10,
        "size_max": 50,
        "target_replicas": 5,
    },
    "bonding_curve": {
        "p_min": 0.5,
        "p_max": 2.0,
        "rho0": 0.75,
        "k": 15.0,
    },
    "miner_behavior": {"beta_leave": 2.0},
    "repair": {
        "base_reward": 0,
        "alpha": 0,
        "beta_repair": 0,
    },
    "endowment": {"initial": 10_000, "yield_rate": 0.008},
}
