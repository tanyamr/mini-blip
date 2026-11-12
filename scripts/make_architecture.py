import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BG = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"
BLUE = "#2a78d6"; BLUE_L = "#dbe9fb"; ORANGE = "#eb6834"; OR_L = "#fde3d8"; VIOLET = "#6250d6"; VI_L = "#e7e4fa"

fig, ax = plt.subplots(figsize=(11, 4.6))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.set_xlim(0, 11.8); ax.set_ylim(0.2, 4.75); ax.axis("off")

def box(x, y, w, h, title, sub, fc, ec, tsize=12):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.6))
    cy = y + h / 2
    if sub:
        ax.text(x + w / 2, cy + 0.13, title, ha="center", va="center", fontsize=tsize, fontweight="bold", color=INK)
        ax.text(x + w / 2, cy - 0.17, sub, ha="center", va="center", fontsize=9.5, color=INK2)
    else:
        ax.text(x + w / 2, cy, title, ha="center", va="center", fontsize=tsize, fontweight="bold", color=INK)

def arrow(x1, y1, x2, y2, dashed=False, color=INK2, label=None, ly=0.12):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6, ls=(0, (4, 3)) if dashed else "-", mutation_scale=14))
    if label:
        ax.text((x1 + x2) / 2, y1 + ly, label, ha="center", va="bottom", fontsize=9, color=INK2)

lanes = [3.6, 2.35, 1.1]
# picture + ViT
box(0.1, 2.0, 1.5, 0.9, "Картинка", None, BLUE_L, BLUE)
arrow(1.6, 2.45, 2.1, 2.45)
box(2.1, 0.75, 1.9, 3.2, "ViT", "патчи → векторы", BLUE_L, BLUE, tsize=13)

mods = [("Текстовый энкодер", "читает подпись"),
        ("Энкодер + cross-attention", "читает подпись, глядя на картинку"),
        ("Декодер + cross-attention", "пишет подпись, глядя на картинку")]
heads = [("ITC", "похожи ли в целом?"), ("ITM", "пара или не пара?"), ("LM", "следующее слово")]
for y, (mt, ms), (ht, hs), dashed, lab in zip(lanes, mods, heads, [False, True, True],
                                             ["вектор картинки", "cross-attention", "cross-attention"]):
    box(5.4, y - 0.4, 3.3, 0.8, mt, ms, OR_L, ORANGE, tsize=11)
    if dashed:
        arrow(4.0, y, 5.4, y, dashed=True, label=lab)
    box(9.3, y - 0.4, 2.3, 0.8, ht, hs, VI_L, VIOLET)
    arrow(8.7, y, 9.3, y)

y0 = 4.3
ax.plot([3.05, 3.05, 10.45], [3.95, y0, y0], color=INK2, lw=1.6)
arrow(10.45, y0, 10.45, lanes[0] + 0.4)
ax.text(7.0, y0 + 0.07, "вектор картинки", ha="center", va="bottom", fontsize=9, color=INK2)
ax.text(0.1, 4.6, "Одна модель, три задачи", fontsize=14, fontweight="bold", color=INK, va="center")
from pathlib import Path
fig.savefig(Path(__file__).resolve().parents[1] / "results/figures/00_architecture.png", dpi=200, bbox_inches="tight", pad_inches=0.2, facecolor=BG)
