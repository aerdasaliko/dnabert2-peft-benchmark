import os
import re
import glob
import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("default")

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid.axis": "y",
})

sns.set_theme(style="whitegrid", context="paper", 
              rc={"axes.axisbelow": False, "grid.alpha": 0.3, "grid.linestyle": "--",
                  "font.size": 12, "axes.titlesize": 14,  "axes.labelsize": 14,
                  "legend.fontsize": 11, "xtick.labelsize": 12, "ytick.labelsize": 12,
                  })

METHOD_ORDER = ["none", "lora", "adalora", "qlora"]
METHOD_COLORS = {
    "none":    "#2ca02c",
    "lora":    "#1f77b4",
    "adalora": "#ff7f0e",
    "qlora":   "#e21a1a",
}

def color_for_methods(method_series):
    return [METHOD_COLORS.get(m, "#333333") for m in method_series]

def ordered_methods_in_df(df, col="method"):
    present = df[col].dropna().unique().tolist()
    ordered = [m for m in METHOD_ORDER if m in present]
    extras = [m for m in present if m not in ordered]
    return ordered + sorted(extras)



LOGS_DIR = "../all-logs/logs-classification"

EVAL_ANALYSIS_DIR = os.path.join(LOGS_DIR, "eval_analysis_outputs")
OVERALL_EVAL_DIR = os.path.join(EVAL_ANALYSIS_DIR, "overall_eval")
TIME_SUMMARY_DIR = os.path.join(LOGS_DIR, "time_summary")
GPU_OVERALL_DIR = os.path.join(LOGS_DIR, "overall_gpu")
GPU_COMPARISON_DIR = os.path.join(LOGS_DIR, "comparisons")
GPU_TIMESERIES_DIR = os.path.join(LOGS_DIR, "timeseries")
GPU_OVERLAYS_DIR = os.path.join(LOGS_DIR, "final_figures")
EXTRA_PLOTS_DIR = os.path.join(LOGS_DIR,"extra_figures")

for d in [
    EVAL_ANALYSIS_DIR, OVERALL_EVAL_DIR, TIME_SUMMARY_DIR, GPU_OVERALL_DIR,
    GPU_COMPARISON_DIR, GPU_TIMESERIES_DIR, GPU_OVERLAYS_DIR, EXTRA_PLOTS_DIR
]:
    os.makedirs(d, exist_ok=True)

# Helpers
def clean_column(series, unit=""):
    return series.astype(str).str.replace(unit, "", regex=False).str.strip().astype(float)

def load_and_clean_gpu_csv(file, skip_rows=5):
    df = pd.read_csv(file)
    df.columns = (
        df.columns.str.strip()
        .str.replace("\n", "", regex=False)
        .str.replace("\r", "", regex=False)
    )

    # required columns
    df["utilization.gpu [%]"] = clean_column(df["utilization.gpu [%]"], "%")
    df["utilization.memory [%]"] = clean_column(df["utilization.memory [%]"], "%")
    if "memory.used [MiB]" in df.columns:
        df["memory.used [MiB]"] = clean_column(df["memory.used [MiB]"], "MiB")
    if "memory.total [MiB]" in df.columns:
        df["memory.total [MiB]"] = clean_column(df["memory.total [MiB]"], "MiB")
    if "power.draw [W]" in df.columns:
        df["power.draw [W]"] = clean_column(df["power.draw [W]"], "W")
    if "temperature.gpu" in df.columns:
        df["temperature.gpu"] = pd.to_numeric(df["temperature.gpu"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    return df.iloc[skip_rows:].copy()

def compute_gpu_metrics(df):
    gpu = df["utilization.gpu [%]"]
    mem_bw = df["utilization.memory [%]"]

    power = (
        df["power.draw [W]"]
        if "power.draw [W]" in df.columns
        else pd.Series(np.nan, index=df.index)
    )

    temp = (
        df["temperature.gpu"]
        if "temperature.gpu" in df.columns
        else pd.Series(np.nan, index=df.index)
    )

    if "memory.used [MiB]" in df.columns:
        memu_used = df["memory.used [MiB]"]
        memu_mean = memu_used.mean()
        memu_peak = memu_used.max()
        memu_min = memu_used.min()
        memu_p95 = memu_used.quantile(0.95)
    else:
        memu_used = pd.Series(np.nan, index=df.index)
        memu_mean = np.nan
        memu_peak = np.nan
        memu_min = np.nan
        memu_p95 = np.nan

    if "memory.total [MiB]" in df.columns:
        memu_total = df["memory.total [MiB]"].median()
    else:
        memu_total = np.nan

    if pd.notna(memu_total) and memu_total > 0:
        memu_mean_pct = (memu_mean / memu_total) * 100
        memu_peak_pct = (memu_peak / memu_total) * 100
    else:
        memu_mean_pct = np.nan
        memu_peak_pct = np.nan

    power_mean = power.mean()
    gpu_mean = gpu.mean()

    if len(df) > 1:
        elapsed_sec = (
            df["timestamp"] - df["timestamp"].iloc[0]
        ).dt.total_seconds()

        gpu_auc = np.trapezoid(
            gpu.fillna(0).values,
            elapsed_sec.values
        )

        mem_bw_auc = np.trapezoid(
            mem_bw.fillna(0).values,
            elapsed_sec.values
        )
    else:
        gpu_auc = np.nan
        mem_bw_auc = np.nan

    return {
        # GPU compute activity
        "gpu_mean": gpu_mean,
        "gpu_max": gpu.max(),
        "gpu_std": gpu.std(),

        # Memory bandwidth activity
        "mem_bw_mean": mem_bw.mean(),
        "mem_bw_max": mem_bw.max(),
        "mem_bw_std": mem_bw.std(),

        # Actual memory consumption
        "memu_mean_mib": memu_mean,
        "memu_peak_mib": memu_peak,
        "memu_min_mib": memu_min,
        "memu_p95_mib": memu_p95,

        # Actual percentage of memory capacity
        "memu_mean_pct": memu_mean_pct,
        "memu_peak_pct": memu_peak_pct,

        # GPU memory capacity
        "memu_total_mib": memu_total,

        # Power / temperature
        "power_mean": power_mean,
        "power_max": power.max(),
        "power_std": power.std(),

        "temp_mean": temp.mean(),
        "temp_max": temp.max(),

        "gpu_efficiency": (
            gpu_mean / power_mean
            if pd.notna(power_mean) and power_mean > 0
            else np.nan
        ),

        "gpu_auc": gpu_auc,
        "mem_bw_auc": mem_bw_auc,
    }

def is_dominated(row, df):
    return any(
        (other["gpu_efficiency"] >= row["gpu_efficiency"] and
         other["runtime_min"] <= row["runtime_min"] and
         (other["gpu_efficiency"] > row["gpu_efficiency"] or
          other["runtime_min"] < row["runtime_min"]))
        for _, other in df.iterrows()
    )

def mean_std_stack(runs):
    if len(runs) == 0:
        return None, None
    min_len = min(len(r) for r in runs)
    stacked = np.stack([r[:min_len] for r in runs], axis=1)
    return stacked.mean(axis=1), stacked.std(axis=1)

METHOD_NAME_MAP = {
    "none": "Full FT",
    "lora": "LoRA",
    "qlora": "QLoRA",
    "adalora": "AdaLoRA",
}

def pretty_method(m):
    return METHOD_NAME_MAP.get(m, m)


# 1) EVAL SUMMARY

print("Running eval summary...")
FILE_PATTERN = r"eval_(.+?)_(none|lora|adalora|qlora)_(\d{8}_\d{6})"

METRIC_LABELS = {
    "eval_accuracy": "Accuracy",
    "eval_f1": "F1 Score",
    "eval_precision": "Precision",
    "eval_recall": "Recall",
    "eval_matthews_correlation": "MCC",
    "eval_roc_auc": "ROC-AUC",
    "eval_pr_auc": "PR-AUC",
    "eval_loss": "Loss"
}
METRIC_ORDER = ["Accuracy", "F1 Score", "Precision", "Recall", "ROC-AUC", "PR-AUC"]
PLOT_METRIC_RAW = [k for k, v in METRIC_LABELS.items() if v in METRIC_ORDER]

eval_records = []
for fname in os.listdir(LOGS_DIR):
    m = re.match(FILE_PATTERN, fname)
    if not m:
        continue
    dataset, method, timestamp = m.groups()
    path = os.path.join(LOGS_DIR, fname)
    try:
        with open(path, "r") as f:
            data = json.load(f)
        eval_records.append({"dataset": dataset, "method": method, "timestamp": timestamp, **data})
    except Exception as e:
        print(f"Skipping {fname}: {e}")

eval_df = pd.DataFrame(eval_records)
if eval_df.empty:
    raise ValueError("No valid eval files found in ../logs")
eval_df["method_pretty"] = eval_df["method"].map(METHOD_NAME_MAP)
eval_df["timestamp"] = pd.to_datetime(eval_df["timestamp"], format="%Y%m%d_%H%M%S", errors="coerce")
for raw in METRIC_LABELS.keys():
    if raw not in eval_df.columns:
        eval_df[raw] = np.nan
eval_df.to_csv(os.path.join(EVAL_ANALYSIS_DIR, "all_results.csv"), index=False)

# per-dataset metrics comparison
for dataset in eval_df["dataset"].dropna().unique():
    subset = eval_df[eval_df["dataset"] == dataset]
    agg = subset.groupby("method")[PLOT_METRIC_RAW].mean(numeric_only=True).reset_index()

    method_order = [m for m in METHOD_ORDER if m in agg["method"].tolist()] + \
                   sorted([m for m in agg["method"].tolist() if m not in METHOD_ORDER])
    agg["method"] = pd.Categorical(agg["method"], categories=method_order, ordered=True)
    agg["method_pretty"] = agg["method"].map(METHOD_NAME_MAP)
    agg = agg.sort_values("method")

    melted = agg.melt(id_vars=["method", "method_pretty"], var_name="metric", value_name="value")
    melted["metric"] = melted["metric"].map(METRIC_LABELS)
    melted["metric"] = pd.Categorical(melted["metric"], categories=METRIC_ORDER, ordered=True)

    plt.figure(figsize=(12, 5))
    sns.barplot(
        data=melted, 
        x="metric", y="value", hue="method_pretty", 
        hue_order=[METHOD_NAME_MAP.get(m, m) for m in method_order], 
        palette={METHOD_NAME_MAP.get(m, m): METHOD_COLORS.get(m, "#333333") for m in method_order},
        saturation=1
        )

    plt.title(f"{dataset}: Method Comparison Across Metrics")
    plt.xlabel("Method")
    plt.ylabel("Value")
    plt.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_ANALYSIS_DIR, f"{dataset}_metrics_comparison.png"))
    plt.close()

dataset_summary = eval_df.groupby(["dataset", "method"])[list(METRIC_LABELS.keys())].mean(numeric_only=True).round(4)
dataset_summary.to_csv(os.path.join(EVAL_ANALYSIS_DIR, "summary_by_dataset_method.csv"))

overall_eval = eval_df.groupby("method")[PLOT_METRIC_RAW].mean(numeric_only=True).reset_index()
method_order_eval = [m for m in METHOD_ORDER if m in overall_eval["method"].tolist()] + \
                    sorted([m for m in overall_eval["method"].tolist() if m not in METHOD_ORDER])
overall_eval["method"] = pd.Categorical(overall_eval["method"], categories=method_order_eval, ordered=True)
overall_eval["method_pretty"] = overall_eval["method"].map(METHOD_NAME_MAP)
overall_eval = overall_eval.sort_values("method")
overall_eval.to_csv(os.path.join(OVERALL_EVAL_DIR, "overall_summary.csv"), index=False)

melted_overall = overall_eval.melt(id_vars=["method", "method_pretty"], var_name="metric", value_name="value")
melted_overall["metric"] = melted_overall["metric"].map(METRIC_LABELS)
melted_overall["metric"] = pd.Categorical(melted_overall["metric"], categories=METRIC_ORDER, ordered=True)

plt.figure(figsize=(12, 5))
sns.barplot(
    data=melted_overall, 
    x="metric", y="value", 
    hue="method_pretty", 
    hue_order=[METHOD_NAME_MAP.get(m, m) for m in method_order_eval], 
    palette={METHOD_NAME_MAP.get(m, m): METHOD_COLORS.get(m, "#333333") for m in method_order_eval},
    saturation=1
)
plt.title("Overall Method Comparison (Mean Across Datasets)")
plt.xlabel("Metric")
plt.ylabel("Value")
# plt.xticks(rotation=45)
plt.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
plt.tight_layout()
plt.savefig(os.path.join(OVERALL_EVAL_DIR, "overall_metrics_comparison.png"))
plt.close()

for raw_metric, pretty_name in METRIC_LABELS.items():
    if raw_metric not in overall_eval.columns:
        continue
    plt.figure(figsize=(6, 4))
    sns.barplot(data=overall_eval, 
                x="method_pretty", 
                y=raw_metric,
                order=[METHOD_NAME_MAP.get(m, m) for m in method_order_eval],
                palette=[METHOD_COLORS.get(m, "#333333") for m in method_order_eval],
                saturation=1)
    plt.title(f"{pretty_name} (Mean Across Datasets)")
    plt.xlabel("Method")
    plt.ylabel(f"{pretty_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OVERALL_EVAL_DIR, f"{pretty_name.lower().replace(' ', '_')}.png"))
    plt.close()


# 2) GPU SUMMARY
print("Running GPU summary...")
gpu_csv_pattern = "gpu_*_*_[0-9]*.csv"
gpu_files = glob.glob(os.path.join(LOGS_DIR, gpu_csv_pattern))

gpu_records = []
time_series_data = {}

for file in gpu_files:
    base = os.path.basename(file).replace(".csv", "")
    parts = base.split("_")
    method = parts[-3]
    dataset = "_".join(parts[1:-3])

    gdf = load_and_clean_gpu_csv(file, skip_rows=5)
    metrics = compute_gpu_metrics(gdf)

    runtime_min = (gdf["timestamp"].iloc[-1] - gdf["timestamp"].iloc[0]).total_seconds() / 60
    gpu_auc = gdf["utilization.gpu [%]"].sum()
    power_mean = metrics["power_mean"]
    energy = (power_mean * runtime_min * 60.0) / 1000

    gpu_records.append({
        "dataset": dataset,
        "method": method,
        "runtime_min": runtime_min,
        "gpu_auc": gpu_auc,
        "energy": energy,
        **metrics
    })

    time_series_data[(dataset, method)] = gdf

summary_df = pd.DataFrame(gpu_records)
summary_df.to_csv(os.path.join(LOGS_DIR, "gpu_summary.csv"), index=False)
summary_df["method_pretty"] = summary_df["method"].map(METHOD_NAME_MAP)

runtime_summary_stats = summary_df.groupby("method")["runtime_min"].agg(["mean", "std", "min", "max"]).reset_index()
runtime_summary_stats.columns = ["method", "runtime_mean", "runtime_std", "runtime_min", "runtime_max"]
runtime_summary_stats["method"] = pd.Categorical(runtime_summary_stats["method"],
                                                 categories=ordered_methods_in_df(runtime_summary_stats),
                                                 ordered=True)
runtime_summary_stats = runtime_summary_stats.sort_values("method")
runtime_summary_stats.to_csv(os.path.join(LOGS_DIR, "runtime_summary.csv"), index=False)

runtime_summary = summary_df.groupby("method")["runtime_min"].mean().reset_index()
runtime_summary["method"] = pd.Categorical(runtime_summary["method"],
                                           categories=ordered_methods_in_df(runtime_summary),
                                           ordered=True)
runtime_summary = runtime_summary.sort_values("method")

method_means = summary_df.groupby("method").mean(numeric_only=True).reset_index()
method_means["method"] = pd.Categorical(method_means["method"],
                                        categories=ordered_methods_in_df(method_means),
                                        ordered=True)
method_means = method_means.sort_values("method")

pareto_df = summary_df.groupby("method").agg({"gpu_efficiency": "mean", "runtime_min": "mean"}).reset_index()
pareto_df["pareto"] = ~pareto_df.apply(lambda r: is_dominated(r, pareto_df), axis=1)

auc_summary = summary_df.groupby("method")["gpu_auc"].mean().reset_index()
auc_summary["method"] = pd.Categorical(auc_summary["method"],
                                       categories=ordered_methods_in_df(auc_summary),
                                       ordered=True)
auc_summary = auc_summary.sort_values("method")

memu_stats = (
    summary_df.groupby("method")
    .agg(
        avg_of_run_avg_mib=("memu_mean_mib", "mean"),
        avg_of_run_peak_mib=("memu_peak_mib", "mean"),
        avg_of_run_p95_mib=("memu_p95_mib", "mean"),
        std_peak_mib=("memu_peak_mib", "std"),
    )
    .reset_index()
)

memu_stats["method"] = pd.Categorical(memu_stats["method"], categories=METHOD_ORDER, ordered=True)
memu_stats = memu_stats.sort_values("method")
memu_stats.to_csv(os.path.join(GPU_OVERALL_DIR, "memu_stats_by_method.csv"), index=False)

metrics_4 = ["gpu_mean", "mem_bw_mean", "power_mean", "temp_mean"]
titles_4 = ["GPU Utilization (%)", "Memory Usage (MiB/%)", "Power Draw (W)", "Temperature (°C)"]

for dataset in summary_df["dataset"].dropna().unique():
    subset = summary_df[summary_df["dataset"] == dataset].copy()
    subset["method"] = pd.Categorical(subset["method"], categories=ordered_methods_in_df(subset), ordered=True)
    subset = subset.sort_values("method")
    colors = color_for_methods(subset["method"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for i, (metric, title) in enumerate(zip(metrics_4, titles_4)):
        ax = axes[i // 2, i % 2]
        if metric in subset.columns:
            ax.bar(subset["method"].astype(str), subset[metric], color=colors)
        ax.set_title(title)
        ax.set_xlabel("Method")
        ax.set_ylabel(title)
    fig.suptitle(f"{dataset}: Method Comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(GPU_COMPARISON_DIR, f"{dataset}_comparison.png"))
    plt.close()

# dataset timeseries 2x2
for dataset in summary_df["dataset"].dropna().unique():
    methods_here = summary_df[summary_df["dataset"] == dataset]["method"].unique().tolist()
    methods_here = [m for m in METHOD_ORDER if m in methods_here] + sorted([m for m in methods_here if m not in METHOD_ORDER])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for method in methods_here:
        dfm = time_series_data[(dataset, method)]
        c = METHOD_COLORS.get(method, "#333333")
        axes[0, 0].plot(dfm["utilization.gpu [%]"].values, label=METHOD_NAME_MAP.get(method, method), color=c)
        axes[0, 1].plot(dfm["utilization.memory [%]"].values, label=method, color=c)
        if "power.draw [W]" in dfm.columns:
            axes[1, 0].plot(dfm["power.draw [W]"].values, label=METHOD_NAME_MAP.get(method, method), color=c)
        if "temperature.gpu" in dfm.columns:
            axes[1, 1].plot(dfm["temperature.gpu"].values, label=METHOD_NAME_MAP.get(method, method), color=c)

    axes[0, 0].set_title("GPU Utilization (%)")
    axes[0, 1].set_title("Memory Utilization (%)")
    axes[1, 0].set_title("Power Draw (W)")
    axes[1, 1].set_title("Temperature (°C)")
    for ax in axes.ravel():
        ax.set_xlabel("Time")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="lower center",
            ncol=max(1, len(labels)),
            frameon=False,
            bbox_to_anchor=(0.5, -0.02)
        )
        fig.tight_layout(rect=[0, 0.08, 1, 0.96]) 

    fig.suptitle(f"{dataset}: GPU Metrics Over Time", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(GPU_TIMESERIES_DIR, f"{dataset}_combined.png"))
    plt.close()

for dataset in summary_df["dataset"].dropna().unique():
    subset = summary_df[summary_df["dataset"] == dataset].copy()
    subset["method"] = pd.Categorical(subset["method"], categories=ordered_methods_in_df(subset), ordered=True)
    subset["method_pretty"] = subset["method"].map(METHOD_NAME_MAP)
    subset = subset.sort_values("method")
    colors = color_for_methods(subset["method"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, title in zip(axes, ["gpu_mean", "mem_bw_mean"], ["GPU Utilization (%)", "Memory Utilization (%)"]):
        ax.bar(subset["method_pretty"], subset[metric], color=colors)
        ax.set_title(title)
        ax.set_xlabel("Method")
        ax.set_ylabel("%")
    fig.suptitle(f"{dataset}: Method Comparison")
    fig.tight_layout()
    plt.savefig(os.path.join(GPU_COMPARISON_DIR, f"{dataset}_comparison_v2.png"))
    plt.close()

for dataset in summary_df["dataset"].dropna().unique():
    methods_here = summary_df[summary_df["dataset"] == dataset]["method"].unique().tolist()
    methods_here = [m for m in METHOD_ORDER if m in methods_here] + sorted([m for m in methods_here if m not in METHOD_ORDER])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for method in methods_here:
        dfm = time_series_data[(dataset, method)]
        pretty = METHOD_NAME_MAP.get(method, method)
        c = METHOD_COLORS.get(method, "#333333")
        axes[0].plot(dfm["utilization.gpu [%]"].values, label=pretty, color=c)
        axes[1].plot(dfm["utilization.memory [%]"].values, label=pretty, color=c)

    axes[0].set_title("GPU Utilization (%)")
    axes[1].set_title("Memory Utilization (%)")
    for ax in axes:
        ax.set_xlabel("Time")
        ax.set_ylabel("%")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="lower center",
            ncol=max(1, len(labels)),
            frameon=False,
            bbox_to_anchor=(0.5, -0.02)
        )
        fig.tight_layout(rect=[0, 0.10, 1, 0.95])

    fig.suptitle(f"{dataset}: Time Series")
    fig.tight_layout()
    plt.savefig(os.path.join(GPU_TIMESERIES_DIR, f"{dataset}_timeseries.png"))
    plt.close()

methods_all = ordered_methods_in_df(summary_df)
datasets_all = summary_df["dataset"].dropna().unique().tolist()
n = len(methods_all)
ncols = int(math.ceil(math.sqrt(n)))
nrows = int(math.ceil(n / ncols))

def method_runs(metric_col, method):
    runs = []
    for dataset in datasets_all:
        if (dataset, method) in time_series_data:
            runs.append(time_series_data[(dataset, method)][metric_col].reset_index(drop=True).values)
    return runs

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axes = np.array(axes).reshape(-1)
for i, method in enumerate(methods_all):
    ax = axes[i]
    runs = method_runs("utilization.gpu [%]", method)
    mean, std = mean_std_stack(runs)
    if mean is None:
        ax.axis("off")
        continue
    x = np.arange(len(mean))
    c = METHOD_COLORS.get(method, "#333333")
    ax.plot(x, mean, label=method, color=c)
    ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=c)
    ax.set_title(METHOD_NAME_MAP.get(method, method))
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Mean GPU Utilization (%)")
for j in range(n, len(axes)):
    axes[j].axis("off")
fig.suptitle("Mean GPU Utilization Across Datasets", y=1.02)
fig.tight_layout()
plt.savefig(os.path.join(GPU_OVERLAYS_DIR, "gpu_subplots.png"), bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axes = np.array(axes).reshape(-1)
for i, method in enumerate(methods_all):
    ax = axes[i]
    runs = method_runs("utilization.memory [%]", method)
    mean, std = mean_std_stack(runs)
    if mean is None:
        ax.axis("off")
        continue
    x = np.arange(len(mean))
    c = METHOD_COLORS.get(method, "#333333")
    ax.plot(x, mean, label=method, color=c)
    ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=c)
    ax.set_title(method)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Mean Memory Utilization (%)")
for j in range(n, len(axes)):
    axes[j].axis("off")
fig.suptitle("Mean Memory Utilization Across Datasets", y=1.02)
fig.tight_layout()
plt.savefig(os.path.join(GPU_OVERLAYS_DIR, "memory_subplots.png"), bbox_inches="tight")
plt.close()

# overlays
def overlay(metric_col, title, filename):
    plt.figure(figsize=(12, 6))
    for method in methods_all:
        runs = method_runs(metric_col, method)
        mean, std = mean_std_stack(runs)
        if mean is None:
            continue
        x = np.arange(len(mean))
        c = METHOD_COLORS.get(method, "#333333")
        plt.plot(x, mean, label=METHOD_NAME_MAP.get(method, method), color=c)
        plt.fill_between(x, mean - std, mean + std, alpha=0.15, color=c)
    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Mean Memory Allocated (MiB)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(GPU_OVERLAYS_DIR, filename), dpi=300)
    plt.close()

overlay("utilization.gpu [%]", "Mean GPU Utilization Across Datasets", "gpu_overlay.png")
overlay("utilization.memory [%]", "Mean Memory Utilization Across Datasets", "memory_overlay.png")
overlay("memory.used [MiB]", "Mean Memory Usage Over Time", "memory_used_overlay.png")

MEAN_LABELS = {
    "gpu_mean": "Mean GPU Utilization (%)",
    "mem_bw_mean": "Mean Memory Utilization (%)",
    "power_mean": "Mean Power Draw (W)",
    "temp_mean": "Mean Temperature (°C)",
    "gpu_efficiency": "GPU Efficiency",
}
# overall GPU figures (from summarize_gpu.py)
for metric, pretty_name in MEAN_LABELS.items():
    if metric not in method_means.columns:
        continue
    plt.figure()
    mdf = method_means.sort_values("method")
    mdf["method_pretty"] = mdf["method"].map(METHOD_NAME_MAP)
    plt.bar(mdf["method_pretty"], mdf[metric], color=color_for_methods(mdf["method"]))
    plt.title(f"Overall {pretty_name} Across Datasets")
    plt.xlabel("Method")
    plt.ylabel(f"{pretty_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(GPU_OVERALL_DIR, f"{metric}_overall.png"))
    plt.close()

plt.figure()
plt.bar(runtime_summary["method"].astype(str), runtime_summary["runtime_min"], color=color_for_methods(runtime_summary["method"]))
plt.title("Average Runtime per Method")
plt.xlabel("Method")
plt.ylabel("Runtime (minutes)")
plt.tight_layout()
plt.savefig(os.path.join(GPU_OVERALL_DIR, "runtime_per_method.png"))
plt.close()

plt.figure()
for _, row in pareto_df.iterrows():
    m = row["method"]
    c = METHOD_COLORS.get(m, "#333333")
    marker = "o" if row["pareto"] else "x"
    plt.scatter(row["runtime_min"], row["gpu_efficiency"], marker=marker, color=c)
    plt.text(row["runtime_min"], row["gpu_efficiency"], m, color=c)
plt.xlabel("Runtime (min)")
plt.ylabel("GPU Efficiency")
plt.title("Pareto Frontier (Efficiency vs Runtime)")
plt.tight_layout()
plt.savefig(os.path.join(GPU_OVERALL_DIR, "pareto_frontier.png"))
plt.close()

plt.figure()
data = [summary_df[summary_df["method"] == m]["gpu_mean"].dropna().values for m in methods_all]
bp = plt.boxplot(data, tick_labels=[METHOD_NAME_MAP.get(m, m) for m in methods_all], 
                 patch_artist=True, medianprops=dict(color="black"))
for patch, m in zip(bp["boxes"], methods_all):
    patch.set_facecolor(METHOD_COLORS.get(m, "#000000"))
plt.title("GPU Utilization Distribution per Method")
plt.ylabel("GPU Utilization (%)")
plt.tight_layout()
plt.savefig(os.path.join(GPU_OVERALL_DIR, "gpu_boxplot.png"))
plt.close()

plt.figure()
data = [summary_df[summary_df["method"] == m]["mem_bw_mean"].dropna().values for m in methods_all]
bp = plt.boxplot(data, tick_labels=[METHOD_NAME_MAP.get(m, m) for m in methods_all], 
                 patch_artist=True, medianprops=dict(color="black"))
for patch, m in zip(bp["boxes"], methods_all):
    patch.set_facecolor(METHOD_COLORS.get(m, "#333333"))
plt.title("Memory Usage Across Datasets per Method")
plt.ylabel("Mean Memory Usage (MiB/%)")
plt.tight_layout()
plt.savefig(os.path.join(GPU_OVERALL_DIR, "memory_boxplot_overall.png"))
plt.close()

plt.figure()
auc_df = auc_summary.sort_values("method")
auc_df["method_pretty"] = auc_df["method"].map(METHOD_NAME_MAP)
plt.bar(auc_df["method_pretty"], auc_df["gpu_auc"], color=color_for_methods(auc_df["method"]))
plt.title("Total GPU Work (AUC) per Method")
plt.ylabel("Sum of GPU Utilization")
plt.tight_layout()
plt.savefig(os.path.join(GPU_OVERALL_DIR, "gpu_auc.png"))
plt.close()

# 3) TIME SUMMARY

print("Running training-time summary...")
train_times_csv = os.path.join(LOGS_DIR, "train_times.csv")
time_df = pd.read_csv(train_times_csv)
time_df["method_pretty"] = time_df["method"].map(METHOD_NAME_MAP)
time_df["dataset"] = time_df["run_name"].apply(lambda x: x.rsplit("_", 1)[0])
time_df["elapsed_sec"] = pd.to_numeric(time_df["elapsed_sec"], errors="coerce") / 60

mean_df = time_df.groupby(["dataset", "method"])["elapsed_sec"].mean().reset_index()
pivot_mean = mean_df.pivot(index="dataset", columns="method", values="elapsed_sec")
pivot_mean = pivot_mean.reindex(columns=[m for m in METHOD_ORDER if m in pivot_mean.columns] +
                                [c for c in pivot_mean.columns if c not in METHOD_ORDER])

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(pivot_mean.index))
cols = pivot_mean.columns.tolist()
w = 0.8 / max(len(cols), 1)
for i, m in enumerate(cols):
    ax.bar(x + (i - (len(cols)-1)/2)*w, pivot_mean[m].values, width=w,
           label=METHOD_NAME_MAP.get(m, m), color=METHOD_COLORS.get(m, "#333333"))
ax.set_xticks(x)
ax.set_xticklabels(pivot_mean.index)
ax.set_title("Mean Runtime per Dataset and Method")
ax.set_ylabel("Elapsed Time (min)")
plt.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
plt.tight_layout()
plt.savefig(os.path.join(TIME_SUMMARY_DIR, "mean_runtime_by_dataset_method.png"), dpi=300)
plt.close()

pivot_raw = time_df.pivot_table(index="dataset", columns="method", values="elapsed_sec", aggfunc="mean")
pivot_raw = pivot_raw.reindex(columns=[m for m in METHOD_ORDER if m in pivot_raw.columns] +
                              [c for c in pivot_raw.columns if c not in METHOD_ORDER])

plt.figure(figsize=(10, 6))
data = [time_df[time_df["method"] == m]["elapsed_sec"].dropna().values for m in ordered_methods_in_df(time_df)]
labels = ordered_methods_in_df(time_df)
bp = plt.boxplot(data, tick_labels=[METHOD_NAME_MAP.get(m, m) for m in ordered_methods_in_df(time_df)], 
                 patch_artist=True, medianprops=dict(color="black"))
for patch, m in zip(bp["boxes"], labels):
    patch.set_facecolor(METHOD_COLORS.get(m, "#333333"))
plt.title("Runtime Distribution by Method")
plt.ylabel("Elapsed Time (min)")
plt.tight_layout()
plt.savefig(os.path.join(TIME_SUMMARY_DIR, "runtime_distribution_by_method.png"), dpi=300)
plt.close()

method_summary = time_df.groupby("method")["elapsed_sec"].agg(["mean", "median", "std", "count"]).sort_values("mean")
method_summary.to_csv(os.path.join(TIME_SUMMARY_DIR, "method_summary_stats.csv"))

plt.figure(figsize=(8, 5))
ms = method_summary.reset_index()
ms["method"] = pd.Categorical(ms["method"], categories=ordered_methods_in_df(ms), ordered=True)
ms = ms.sort_values("method")
ms["method_pretty"] = ms["method"].map(METHOD_NAME_MAP)
plt.bar(ms["method_pretty"], ms["mean"], color=color_for_methods(ms["method"]))
plt.title("Overall Mean Runtime by Method")
plt.ylabel("Runtime (minutes)")
plt.tight_layout()
plt.savefig(os.path.join(TIME_SUMMARY_DIR, "overall_mean_runtime_by_method.png"), dpi=300)
plt.close()

baseline = (time_df[time_df["method"] == "none"].groupby("dataset")["elapsed_sec"].mean().rename("baseline_sec").reset_index())
time_df = time_df.merge(baseline, on="dataset", how="left")
time_df["speedup"] = time_df["baseline_sec"] / time_df["elapsed_sec"]

speedup_table = time_df.pivot_table(index="dataset", columns="method", values="speedup", aggfunc="mean")
speedup_table = speedup_table.reindex(columns=[m for m in METHOD_ORDER if m in speedup_table.columns] +
                                      [c for c in speedup_table.columns if c not in METHOD_ORDER])

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(speedup_table.index))
cols = speedup_table.columns.tolist()
w = 0.8 / max(len(cols), 1)
for i, m in enumerate(cols):
    ax.bar(x + (i - (len(cols)-1)/2)*w, speedup_table[m].values, width=w,
           label=METHOD_NAME_MAP.get(m, m), color=METHOD_COLORS.get(m, "#333333"))
ax.set_xticks(x)
ax.set_xticklabels(speedup_table.index)
ax.set_title("Speedup vs Baseline (none) per Dataset and Method")
ax.set_ylabel("Speedup (x faster than baseline)")
plt.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
plt.tight_layout()
plt.savefig(os.path.join(TIME_SUMMARY_DIR, "speedup_by_dataset_method.png"), dpi=300)
plt.close()

method_speedup = time_df.groupby("method")["speedup"].agg(["mean", "median", "std", "count"]).sort_values("mean", ascending=False)
method_speedup.to_csv(os.path.join(TIME_SUMMARY_DIR, "method_speedup_stats.csv"))

plt.figure(figsize=(8, 5))
msu = method_speedup.reset_index()
msu["method"] = pd.Categorical(msu["method"], categories=ordered_methods_in_df(msu), ordered=True)
msu = msu.sort_values("method")
msu["method_pretty"] = msu["method"].map(METHOD_NAME_MAP)
plt.bar(msu["method_pretty"], msu["mean"], color=color_for_methods(msu["method"]))
plt.title("Average Speedup vs Baseline by Method")
plt.ylabel("Speedup (higher is better)")
plt.tight_layout()
plt.savefig(os.path.join(TIME_SUMMARY_DIR, "overall_speedup_by_method.png"), dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
methods = ["lora", "qlora", "adalora"]
data = [time_df[time_df["method"] == m]["speedup"].dropna().values for m in methods]
labels = methods
bp = plt.boxplot(data, tick_labels=[METHOD_NAME_MAP.get(m, m) for m in methods], 
                 patch_artist=True, medianprops=dict(color="black"))
for patch, m in zip(bp["boxes"], labels):
    patch.set_facecolor(METHOD_COLORS.get(m, "#333333"))
plt.title("Speedup Distribution by Method (vs none)")
plt.ylabel("Speedup")
plt.tight_layout()
plt.savefig(os.path.join(TIME_SUMMARY_DIR, "speedup_distribution_by_method.png"), dpi=300)
plt.close()

# 4) CROSS ANALYSIS

print("Running cross-analysis...")
resource_df = pd.read_csv(os.path.join(LOGS_DIR, "gpu_summary.csv"))
runtime_df = pd.read_csv(os.path.join(LOGS_DIR, "runtime_summary.csv"))
eval_overall_df = pd.read_csv(os.path.join(OVERALL_EVAL_DIR, "overall_summary.csv"))

merged = eval_overall_df.merge(runtime_df, on="method")
usage = resource_df.groupby("method").mean(numeric_only=True).reset_index()
merged = merged.merge(usage, on="method")

merged["method"] = pd.Categorical(merged["method"], categories=ordered_methods_in_df(merged), ordered=True)
merged = merged.sort_values("method")
methods = merged["method"].astype(str).tolist()
merged["method_pretty"] = merged["method"].map(METHOD_NAME_MAP)

fig, axs = plt.subplots(2, 2, figsize=(11, 8))

# A F1 Score
axs[0, 0].bar(merged["method_pretty"], merged["eval_f1"], color=color_for_methods(methods))
axs[0, 0].set_title("A. F1 Score")
axs[0, 0].set_ylim(0, 1)
axs[0, 0].grid(axis="y", alpha=0.3)

# B Runtime with error
axs[0, 1].bar(merged["method_pretty"], merged["runtime_mean"], yerr=merged["runtime_std"], capsize=5,
              color=color_for_methods(methods))
axs[0, 1].set_title("B. Training Time (mean ± std)")
axs[0, 1].set_ylabel("Minutes")
axs[0, 1].grid(axis="y", alpha=0.3)

# C Pareto
axs[1, 0].scatter(merged["runtime_mean"], merged["eval_f1"],
                  c=[METHOD_COLORS.get(m, "#333333") for m in methods])
for i, m in enumerate(methods):
    axs[1, 0].annotate(m, (merged["runtime_mean"].iloc[i], merged["eval_f1"].iloc[i]))
axs[1, 0].set_title("C. F1 vs Runtime (Pareto View)")
axs[1, 0].set_xlabel("Runtime (s)")
axs[1, 0].set_ylabel("F1")
axs[1, 0].grid(alpha=0.3)

points = merged[["runtime_mean", "eval_f1"]].values
sorted_idx = np.argsort(points[:, 0])
sorted_points = points[sorted_idx]
pareto = []
best_acc = -1
for x, y in sorted_points:
    if y > best_acc:
        pareto.append((x, y))
        best_acc = y
pareto = np.array(pareto)
if len(pareto) > 0:
    axs[1, 0].plot(pareto[:, 0], pareto[:, 1], linestyle="--", color="black", alpha=0.6)

# D Memory Usage vs F1
axs[1, 1].scatter(merged["memu_peak_mib"], merged["eval_accuracy"],
                  c=[METHOD_COLORS.get(m, "#333333") for m in methods])
for i, m in enumerate(methods):
    axs[1, 1].annotate(m, (merged["memu_peak_mib"].iloc[i], merged["eval_accuracy"].iloc[i]))
axs[1, 1].set_title("D. Memory Usage vs F1")
axs[1, 1].set_xlabel("Memory Usage")
axs[1, 1].set_ylabel("F1")
axs[1, 1].grid(alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(EXTRA_PLOTS_DIR, "figure1_main.pdf"))
fig.savefig(os.path.join(EXTRA_PLOTS_DIR, "figure1_main.png"), dpi=300)

plt.figure(figsize=(6, 5))
plt.scatter(merged["gpu_mean"], merged["memu_peak_mib"],
            c=[METHOD_COLORS.get(m, "#333333") for m in methods], s=100)
for i, m in enumerate(methods):
    plt.annotate(m, (merged["gpu_mean"].iloc[i], merged["memu_peak_mib"].iloc[i]))
plt.title("GPU vs Memory Trade-off")
plt.xlabel("GPU Utilization (%)")
plt.ylabel("Memory Usage (MiB)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(EXTRA_PLOTS_DIR, "figure2_gpu_memory_tradeoff.pdf"))
plt.savefig(os.path.join(EXTRA_PLOTS_DIR, "figure2_gpu_memory_tradeoff.png"), dpi=300)

norm = merged.copy()
for col in ["eval_accuracy", "eval_f1", "eval_roc_auc"]:
    denom = (merged[col].max() - merged[col].min())
    norm[col] = 0.0 if denom == 0 else (merged[col] - merged[col].min()) / denom

rt_denom = (merged["runtime_mean"].max() - merged["runtime_mean"].min())
norm["runtime_score"] = 1.0 if rt_denom == 0 else 1 - (merged["runtime_mean"] - merged["runtime_mean"].min()) / rt_denom

memu_denom = merged["memu_peak_mib"].max()

norm["resource_score"] = (
    1.0 if memu_denom == 0
    else 1 - (merged["memu_peak_mib"] / memu_denom)
)

norm["overall_score"] = 0.4 * norm["eval_f1"] + 0.3 * norm["runtime_score"] + 0.3 * norm["resource_score"]

norm["method_pretty"] = norm["method"].map(METHOD_NAME_MAP)
plt.figure(figsize=(6, 6))
plt.bar(norm["method_pretty"], norm["overall_score"], color=color_for_methods(norm["method"].astype(str).tolist()))
plt.title("Overall PEFT Efficiency Score (Normalized)")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(EXTRA_PLOTS_DIR, "figure3_overall_score.pdf"))
plt.savefig(os.path.join(EXTRA_PLOTS_DIR, "figure3_overall_score.png"), dpi=300)

print("All analyses complete.")
print(f"Eval outputs: {EVAL_ANALYSIS_DIR}")
print(f"GPU outputs: {LOGS_DIR}")
print(f"Time outputs: {TIME_SUMMARY_DIR}")
print(f"Paper figures: {EXTRA_PLOTS_DIR}")

# GPU vs Memory Utilization: side-by-side
fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns
ax = axes[0]
ax.set_title("Mean GPU Usage Across Datasets")
mdf = method_means.sort_values("method").copy()
mdf["method_pretty"] = mdf["method"].map(METHOD_NAME_MAP)
ax.bar(
    mdf["method_pretty"],
    mdf["gpu_mean"],
    color=color_for_methods(mdf["method"])
)
ax.set_xlabel("Method")
ax.set_ylabel("Mean GPU Usage(%)")

ax = axes[1]
ax.set_title("Mean Memory Usage Across Datasets")
ax.bar(
    mdf["method_pretty"],
    mdf["mem_bw_mean"],
    color=color_for_methods(mdf["method"])
)
ax.set_xlabel("Method")
ax.set_ylabel("Mean Memory Usage (%)")

plt.tight_layout()
plt.savefig(
    os.path.join(GPU_OVERALL_DIR, "combined_memory_gpu.png"),
    bbox_inches="tight"
)
plt.close()

# Mem Mean and GPU mean Boxplots side-by-side
fig, axes = plt.subplots(1, 2, figsize=(10, 5)) 
ax = axes[0]
sns.boxplot(
    data=summary_df,
    x="method",
    y="mem_bw_mean",
    order=methods_all,
    palette=METHOD_COLORS,
    showfliers=False,
    saturation=1,
    ax=ax
)

ax.set_xticks(range(len(methods_all)))
ax.set_xticklabels([METHOD_NAME_MAP.get(m, m) for m in methods_all])
ax.set_xlabel("Method")
ax.set_ylabel("Mean Memory Utilization (%)")
ax.set_title("Memory Utilization Across Datasets")

ax = axes[1]

sns.boxplot(
    data=summary_df,
    x="method",
    y="gpu_mean",
    order=methods_all,
    palette=METHOD_COLORS,
    showfliers=False,
    saturation=1,
    ax=ax
)

ax.set_xticks(range(len(methods_all)))
ax.set_xticklabels([METHOD_NAME_MAP.get(m, m) for m in methods_all])
ax.set_xlabel("Method")
ax.set_ylabel("Mean GPU Utilization (%)")
ax.set_title("GPU Utilization Across Datasets")

plt.tight_layout()
plt.savefig(
    os.path.join(GPU_OVERALL_DIR, "memory_gpu_boxplots.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# Runtime + Speedup side-by-side
fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns
ax = axes[0]

ax.bar(
    merged["method_pretty"],
    merged["runtime_mean"],
    capsize=5,
    color=color_for_methods(merged["method"])
)

ax.set_title("Mean Runtime Across Datasets")
ax.set_ylabel("Runtime (minutes)")
ax.set_xlabel("Method")
ax.grid(axis="y", alpha=0.3)

ax = axes[1]
msu = method_speedup.reset_index()
msu["method"] = pd.Categorical(
    msu["method"],
    categories=ordered_methods_in_df(msu),
    ordered=True
)
msu = msu.sort_values("method")
msu["method_pretty"] = msu["method"].map(METHOD_NAME_MAP)

ax.bar(
    msu["method_pretty"],
    msu["mean"],
    color=color_for_methods(msu["method"])
)

ax.set_title("Average Speedup Against Full FT")
ax.set_ylabel("Speedup")
ax.set_xlabel("Method")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(
    os.path.join(TIME_SUMMARY_DIR, "runtime_speedup_combined.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# Peak Memory Usage
plt.figure(figsize=(6, 6))
memu_df = (
    summary_df
    .groupby("method")["memu_peak_mib"]
    .agg(["mean", "std", "min", "max"])
    .reset_index()
)
memu_df["method"] = pd.Categorical(
    memu_df["method"],
    categories=METHOD_ORDER,
    ordered=True
)
memu_df = memu_df.sort_values("method")
plt.bar(
    memu_df["method"].map(METHOD_NAME_MAP),
    memu_df["mean"],
    capsize=5,
    color=color_for_methods(memu_df["method"])
)

plt.ylabel("Allocated Memory (MiB)")
plt.xlabel("Method")
plt.title("Average Peak Memory Usage Across Datasets")

plt.tight_layout()
plt.savefig(
    os.path.join(
        GPU_OVERALL_DIR,
        "peak_memu_usage.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# Peak Memory Usage boxplot
plt.figure(figsize=(6, 6))
sns.boxplot(
    data=summary_df,
    x="method",
    y="memu_peak_mib",
    order=methods_all,
    palette=METHOD_COLORS,
    showfliers=False,
    saturation=1
)

plt.xticks(
    range(len(methods_all)),
    [METHOD_NAME_MAP.get(m, m) for m in methods_all]
)

plt.xlabel("Method")
plt.ylabel("Allocated Memory (MiB)")
plt.title("Average Peak Memory Usage Across Datasets")

plt.tight_layout()

plt.savefig(
    os.path.join(
        GPU_OVERALL_DIR,
        "peak_memu_boxplot.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# PeakMemory Reduction
memu_by_method = (
    summary_df
    .groupby("method")["memu_peak_mib"]
    .mean()
)

if "none" in memu_by_method:
    baseline_memu = memu_by_method["none"]
    memu_reduction = (
        (baseline_memu - memu_by_method)
        / baseline_memu
        * 100
    )
    memu_reduction = (
        memu_reduction
        .drop(index="none", errors="ignore")
        .reindex([
            m for m in METHOD_ORDER
            if m in memu_reduction.index
        ])
    )
    plt.figure(figsize=(6, 6))
    plt.bar(
        [METHOD_NAME_MAP.get(m, m) for m in memu_reduction.index],
        memu_reduction.values,
        color=color_for_methods(memu_reduction.index)
    )

    plt.axhline(0, linewidth=1)
    plt.ylabel("Peak Memory Reduction vs. Full FT (%)")
    plt.xlabel("Method")
    plt.title("Peak Memory Savings Relative to Full Fine-Tuning")
    plt.tight_layout()
    plt.savefig(
        os.path.join(EXTRA_PLOTS_DIR, "figure_memu_reduction.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    
# Energy
plt.figure(figsize=(6, 6))
edf = method_means.sort_values("method")
edf["method_pretty"] = edf["method"].map(METHOD_NAME_MAP)

plt.bar(
    edf["method_pretty"],
    edf["energy"],
    color=color_for_methods(edf["method"])
)

plt.title("Total Energy Consumption per Method")
plt.xlabel("Method")
plt.ylabel("Energy (kJ)")

plt.tight_layout()
plt.savefig(os.path.join(GPU_OVERALL_DIR, "energy_per_method.png"))
plt.close()