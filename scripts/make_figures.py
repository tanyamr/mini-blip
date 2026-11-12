import json, csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1] / "results"
SRC = str(ROOT)
OUT = str(ROOT / "figures")

BG = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#a3a29c"; GRID = "#e6e5e0"
BLUE = "#2a78d6"; ORANGE = "#eb6834"; AQUA = "#1baf7a"; GRAY = "#b9b8b1"; VIOLET = "#6250d6"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
})

def save(fig, name):
    fig.savefig(f"{OUT}/{name}", dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)

# 1. Training curves
log = json.load(open(f"{SRC}/logs/small_ft_full_trainlog.json"))
ep = [r["epoch"] for r in log]
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8))
for key, lab, c in [("loss_itc", "ITC", BLUE), ("loss_lm", "LM", ORANGE), ("loss_itm", "ITM", AQUA)]:
    ys = [r[key] for r in log]
    a.plot(ep, ys, color=c, lw=2, marker="o", ms=5, mec=BG, mew=1.5)
    a.annotate(lab, (ep[-1], ys[-1]), xytext=(8, 0), textcoords="offset points", va="center", color=INK, fontsize=11)
a.set_title("Лоссы падают"); a.set_xlabel("эпоха"); a.set_xlim(0.7, ep[-1] + 0.9); a.set_ylim(0, None)
ys = [r["val_t2i_R@1"] for r in log]
b.plot(ep, ys, color=BLUE, lw=2, marker="o", ms=5, mec=BG, mew=1.5)
b.annotate(f"{ys[-1]:.1f}%", (ep[-1], ys[-1]), xytext=(0, 10), textcoords="offset points", ha="center", color=INK)
b.set_title("Поиск картинки по тексту, R@1 (валидация)"); b.set_xlabel("эпоха"); b.set_ylim(0, 45)
b.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
fig.tight_layout(w_pad=3)
save(fig, "01_training.png")

# 2. Before / after
rows = list(csv.DictReader(open(f"{SRC}/tables/01_before_after.csv")))
metrics = [("T→I R@1", "Текст → картинка\nR@1"), ("I→T R@1", "Картинка → текст\nR@1"), ("BLEU-4", "BLEU-4"), ("CIDEr", "CIDEr")]
fig, ax = plt.subplots(figsize=(9, 3.8))
w = 0.36
for i, (m, lab) in enumerate(metrics):
    before, after = float(rows[0][m]), float(rows[1][m])
    ax.bar(i - w / 2 - 0.01, before, w, color=GRAY)
    ax.bar(i + w / 2 + 0.01, after, w, color=BLUE)
    ax.text(i - w / 2 - 0.01, before + 0.8, f"{before:g}", ha="center", color=INK2, fontsize=10)
    ax.text(i + w / 2 + 0.01, after + 0.8, f"{after:.1f}", ha="center", color=INK, fontsize=11, fontweight="bold")
ax.set_xticks(range(len(metrics)), [l for _, l in metrics])
ax.tick_params(axis="x", length=0); ax.grid(axis="x", visible=False)
ax.set_title("До и после обучения на 5 000 картинок Flickr30k")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=GRAY, label="до (случайные веса)"), Patch(color=BLUE, label="после обучения")],
          frameon=False, loc="upper left", fontsize=10)
ax.set_ylim(0, 48)
save(fig, "02_before_after.png")

# 3. CapFilt
rows = list(csv.DictReader(open(f"{SRC}/tables/02_capfilt.csv")))
names = ["Без очистки\n(40% шума)", "Только фильтр", "Только captioner", "CapFilt\n(фильтр + captioner)", "Идеал: шума нет"]
colors = [GRAY, BLUE, BLUE, BLUE, VIOLET]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
for ax, (m, title) in zip(axes, [("T→I R@1", "Поиск: текст → картинка, R@1"), ("CIDEr", "Качество подписей, CIDEr")]):
    vals = [float(r[m]) for r in rows]
    y = list(range(len(vals)))[::-1]
    ax.barh(y, vals, 0.62, color=colors)
    for yi, v in zip(y, vals):
        ax.text(v + max(vals) * 0.015, yi, f"{v:.1f}", va="center", color=INK, fontsize=10,
                bbox=dict(boxstyle="square,pad=0.15", fc=BG, ec="none"))
    ax.axvline(vals[0], color=INK2, lw=1, ls=(0, (3, 3)))
    ax.set_title(title); ax.grid(axis="y", visible=False); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(vals) * 1.15)
axes[0].set_yticks(list(range(len(names)))[::-1], names)
fig.tight_layout(w_pad=2)
save(fig, "03_capfilt.png")

# 4. Generation trade-off
rows = list(csv.DictReader(open(f"{SRC}/tables/03_generation.csv")))
fig, ax = plt.subplots(figsize=(8, 4.4))
off = {"greedy": (10, -6, "left"), "beam-3": (10, 6, "left"), "beam-5": (0, 0, "skip"),
       "nucleus p=0.5": (8, 2, "left"), "nucleus p=0.9": (8, 2, "left"), "sampling p=1.0": (-8, 8, "right")}
for r in rows:
    name = r[""]; x, yv = float(r["distinct-2"]), float(r["CIDEr"])
    c = BLUE if name.startswith(("greedy", "beam")) else ORANGE
    ax.scatter(x, yv, s=80, color=c, edgecolor=BG, linewidth=2, zorder=3)
    dx, dy, ha = off[name]
    if ha == "skip": continue
    if name == "beam-3": name = "beam-3 и beam-5"
    ax.annotate(name, (x, yv), xytext=(dx, dy), textcoords="offset points", ha=ha, color=INK, fontsize=10)
ax.set_xlabel("разнообразие подписей (distinct-2) →"); ax.set_ylabel("точность (CIDEr) →")
ax.set_title("Точность против разнообразия")
ax.set_xlim(0, 0.72); ax.set_ylim(0, 48)
ax.text(0.02, 3, "оранжевые — случайный выбор слов (sampling)\nсиние — детерминированный поиск (greedy / beam)", color=INK2, fontsize=9)
save(fig, "04_generation.png")
print("ok")

# 5. Medical (ROCO)

MED = f"{SRC}/tables/med_captioning.csv"
rows = list(csv.DictReader(open(MED)))
labels = {"BLIP zero-shot (исходная)": ("BLIP, без дообучения", GRAY),
          "BLIP, дообучен decoder": ("BLIP, дообучен декодер", BLUE),
          "BLIP, дообучена целиком": ("BLIP, дообучена целиком", BLUE),
          "мини-BLIP Flickr, zero-shot": ("мини-BLIP, без дообучения", GRAY),
          "мини-BLIP Flickr→ROCO": ("мини-BLIP: Flickr → ROCO", ORANGE),
          "мини-BLIP init→ROCO": ("мини-BLIP: сразу на ROCO", ORANGE)}
names = [labels[r[""]][0] for r in rows]; cols = [labels[r[""]][1] for r in rows]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
for ax, (m, title, fmt) in zip(axes, [("CIDEr", "Качество подписей, CIDEr", "{:.1f}"),
                                       ("модальность верна, %", "Верно названа модальность (КТ, МРТ…), %", "{:.0f}%")]):
    vals = [float(r[m]) for r in rows]
    y = list(range(len(vals)))[::-1]
    ax.barh(y, vals, 0.62, color=cols)
    for yi, v in zip(y, vals):
        ax.text(v + max(vals) * 0.015, yi, fmt.format(v), va="center", color=INK, fontsize=10)
    ax.set_title(title); ax.grid(axis="y", visible=False); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(vals) * 1.18)
axes[0].set_yticks(list(range(len(names)))[::-1], names)
fig.tight_layout(w_pad=2)
save(fig, "05_medical.png")
print("ok med")
