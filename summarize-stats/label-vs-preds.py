import pandas as pd
import matplotlib.pyplot as plt

files = [
    "../test-preds/test_predictions_none_k562.csv", 
    "../test-preds/test_predictions_none_hep.csv", 
    "../test-preds/test_predictions_none_sknsh.csv"
]
colors = ["#52b6ac", "#ecb75d", "#ed1e24"]
subplot_text = ["K562 $\it{r}$ = 0.800", "HepG2 $\it{r}$ = 0.807", "SK-N-SH $\it{r}$ = 0.798"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
fig.suptitle("(a) Empirical vs. Predicted Values After Full Fine-Tuning", fontsize=16)

for i, (file, color) in enumerate(zip(files, colors)):
    df = pd.read_csv(file)

    ax = axes[i]
    ax.scatter(df["prediction"], df["label"], color=color, s=0.5, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)


    min_val = int(min(df["label"].min(), df["prediction"].min()))
    max_val = int(max(df["label"].max(), df["prediction"].max()))
    ax.plot([min_val, max_val], [min_val, max_val],
            linestyle='--', color='#b3b3b3', linewidth=3)

    ax.tick_params(axis="both", labelsize=16)

    ax.set_xticks([0, 5])
    ax.set_yticks([0, 5])
    ax.set_xlim(-3, 9)
    ax.set_ylim(-3, 9)

    axes[i].set_xlabel("Predicted activity", fontsize=16)
    if i == 0:
        axes[i].set_ylabel("Empirical activity", fontsize=16)

    ax.text(
        0.93, 0.02, 
        subplot_text[i],
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=16
    )

plt.tight_layout()
plt.savefig("../test-preds/labels-vs-preds-none.png")
plt.close()