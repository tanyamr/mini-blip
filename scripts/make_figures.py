"""Перерисовывает графики для README из results/tables и results/logs."""
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(os.environ.get("RESULTS_DIR", Path(__file__).resolve().parents[1] / "results"))
TABLES, LOGS, FIGS = ROOT / "tables", ROOT / "logs", ROOT / "figures"
FIGS.mkdir(parents=True, exist_ok=True)


def read_csv(name):
    with open(TABLES / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGS / name, dpi=150, facecolor="white")
    plt.close(fig)


# 1. Обучение: лоссы и Recall@1 на валидации
log = json.load(open(LOGS / "small_ft_full_trainlog.json"))
epochs = [r["epoch"] for r in log]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
ax1.plot(epochs, [r["loss_itc"] for r in log], "o-", label="ITC")
ax1.plot(epochs, [r["loss_itm"] for r in log], "o-", label="ITM")
ax1.plot(epochs, [r["loss_lm"] for r in log], "o-", label="LM")
ax1.set_title("Лоссы")
ax1.set_xlabel("эпоха")
ax1.legend()
ax2.plot(epochs, [r["val_t2i_R@1"] for r in log], "o-", color="tab:red")
ax2.set_title("Текст → картинка, R@1 (валидация)")
ax2.set_xlabel("эпоха")
save(fig, "01_training.png")

# 3. CapFilt
rows = read_csv("02_capfilt.csv")
names = ["base", "filter_only", "caption_only", "full", "oracle"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
for ax, m in zip(axes, ["T→I R@1", "CIDEr"]):
    ax.bar(names, [float(r[m]) for r in rows], color=["tab:gray", "tab:blue", "tab:orange", "tab:green", "tab:purple"])
    ax.set_title(m)
    ax.tick_params(axis="x", rotation=20)
save(fig, "02_capfilt.png")

# 4. Стратегии генерации: качество и разнообразие
rows = read_csv("03_generation.csv")
fig, ax = plt.subplots(figsize=(6, 4))
for r in rows:
    ax.scatter(float(r["distinct-2"]), float(r["CIDEr"]))
    dx, dy, ha = {"beam-3": (6, 4, "left"), "beam-5": (-6, -12, "right"), "greedy": (6, -10, "left")}.get(r[""], (6, 3, "left"))
    ax.annotate(r[""], (float(r["distinct-2"]), float(r["CIDEr"])), xytext=(dx, dy), textcoords="offset points", ha=ha)
ax.set_xlabel("distinct-2 (разнообразие)")
ax.set_ylabel("CIDEr (качество)")
ax.set_title("Стратегии генерации")
ax.set_xlim(0, 0.8)
save(fig, "03_generation.png")

# 5. Медицинские снимки (ROCO)
rows = read_csv("med_captioning.csv")
names = [r[""] for r in rows]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, m in zip(axes, ["CIDEr", "модальность верна, %"]):
    ax.barh(names, [float(r[m]) for r in rows])
    ax.set_title(m)
axes[0].invert_yaxis()
save(fig, "04_medical.png")
print("готово:", FIGS)
