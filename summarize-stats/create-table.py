import pandas as pd

df = pd.read_csv("../all-logs/logs-regression/gpu_summary.csv")
df["energy_kJ"] = (df["power_mean"] * df["runtime_min"] * 60) / 1000

# Columns to aggregate
metrics = [
    "runtime_min",
    "vram_mean_mib",
    "vram_peak_mib",
    "power_mean",
    "energy_kJ",
]
aggregated = (
    df.groupby("method")[metrics]
      .mean()
      .reset_index()
)

aggregated.to_csv("../tables/reg-gpu-mem-by-method.csv", index=False)