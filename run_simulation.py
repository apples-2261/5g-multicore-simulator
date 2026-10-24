"""Run all three architecture models from the terminal and save CSV outputs."""
import os
import pandas as pd
from sim_engine import WorkloadConfig, default_architectures, run_comparison, simulate_once

def main():
    os.makedirs("results", exist_ok=True)
    cfg = WorkloadConfig()
    architectures = default_architectures()
    summary, task_tables, distributions = run_comparison(cfg, architectures)

    print("\n=== 5G MULTICORE ARCHITECTURE SIMULATION ===")
    print("All parameters are synthetic model assumptions, not hardware measurements.\n")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    summary.to_csv("results/architecture_comparison.csv", index=False)
    pd.concat([df.assign(Architecture=name) for name, df in task_tables.items()],
              ignore_index=True).to_csv("results/task_breakdown.csv", index=False)

    distribution_rows = []
    for name, values in distributions.items():
        for i, value in enumerate(values):
            distribution_rows.append({"Architecture": name, "Run": i + 1, "Latency (ms)": value})
    pd.DataFrame(distribution_rows).to_csv("results/latency_samples.csv", index=False)
    print("\nSaved:")
    print("  results/architecture_comparison.csv")
    print("  results/task_breakdown.csv")
    print("  results/latency_samples.csv")

if __name__ == "__main__":
    main()
