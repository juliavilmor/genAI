"""Render a schematic of the genAI prefix-decoder architecture to plots/architecture.png.

Standalone: uses only matplotlib + numpy. Run from the repo root:
    python scripts/plotting/plot_architecture.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "plots", "architecture.png")

C_SPECIAL = "#b0b0b0"
C_PROT = "#7fb3d5"
C_DELIM = "#f4d03f"
C_MOL = "#82c99a"
C_EOS = "#e59866"
C_PAD = "#e8e8e8"
C_BLOCK = "#eef3f8"
C_BLOCK_EDGE = "#5d6d7e"
C_LOSS = "#f1948a"

fig = plt.figure(figsize=(15, 12))
gs = fig.add_gridspec(3, 2, height_ratios=[0.7, 1.6, 0.75], hspace=0.55, wspace=0.2)

# ============================================================================
# (1) TOKEN SEQUENCE LAYOUT
# ============================================================================
ax = fig.add_subplot(gs[0, :])
ax.set_xlim(0, 20)
ax.set_ylim(0, 3.4)
ax.axis("off")
ax.set_title("1 · Input sequence  =  prefix (protein)  +  <DELIM>  +  suffix (molecule SMILES)",
             fontsize=12, fontweight="bold", loc="left")

cells = [
    ("<cls>", C_SPECIAL), ("M", C_PROT), ("A", C_PROT), ("R", C_PROT), ("C", C_PROT),
    ("…", C_PROT), ("K", C_PROT), ("<pad>", C_PAD), ("<pad>", C_PAD),
    ("<DELIM>", C_DELIM),
    ("C", C_MOL), ("C", C_MOL), ("O", C_MOL), ("…", C_MOL), ("<eos>", C_EOS),
    ("<pad>", C_PAD), ("<pad>", C_PAD),
]
x, w = 1.0, 1.05
for label, colour in cells:
    ax.add_patch(FancyBboxPatch((x, 1.7), w * 0.92, 0.9, boxstyle="round,pad=0.02",
                                linewidth=1.0, edgecolor="#34495e", facecolor=colour))
    ax.text(x + w * 0.46, 2.15, label, ha="center", va="center", fontsize=9)
    x += w

ax.plot([1.0, 1.0 + 9 * w], [1.55, 1.55], color=C_PROT, lw=2.5)
ax.text(1.0 + 4.5 * w, 1.2, "PREFIX  ·  bidirectional attention  ·  NOT scored in the loss",
        ha="center", fontsize=9, color="#21618c")
ax.plot([1.0 + 10 * w, 1.0 + 17 * w], [1.55, 1.55], color=C_MOL, lw=2.5)
ax.text(1.0 + 13.5 * w, 1.2, "SUFFIX  ·  causal attention  ·  next-token loss here",
        ha="center", fontsize=9, color="#1e8449")
ax.text(1.0 + 9.5 * w, 2.95,
        "<DELIM> (id 24) marks the boundary; the split index is located at run time",
        ha="center", fontsize=8.5, color="#9a7d0a")

vloc = [
    ("special <cls>/<pad>/<eos>/<unk>  ids 0-3", C_SPECIAL),
    ("20 amino acids  ids 4-23", C_PROT),
    ("<DELIM>  id 24", C_DELIM),
    ("46 SMILES regex tokens  ids 25-70", C_MOL),
]
lx = 1.0
for label, colour in vloc:
    ax.add_patch(FancyBboxPatch((lx, 0.2), 0.35, 0.35, boxstyle="round,pad=0.02",
                                edgecolor="#34495e", facecolor=colour, linewidth=1.0))
    ax.text(lx + 0.5, 0.37, label, ha="left", va="center", fontsize=8)
    lx += 4.9

# ============================================================================
# (2) MODEL STACK  (top -> bottom flow)
# ============================================================================
ax = fig.add_subplot(gs[1, 0])
ax.set_xlim(0, 10)
ax.set_ylim(0, 20)
ax.axis("off")
ax.set_title("2 · Model  (decoder-only Transformer)", fontsize=12, fontweight="bold", loc="left")

CX = 5.0


def box(y, h, text, fc=C_BLOCK, fs=8.7, w=8.4, ec=C_BLOCK_EDGE, x=None):
    x = CX - w / 2 if x is None else x
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                linewidth=1.3, edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def down(y_top, y_bot, x=CX):
    ax.add_patch(FancyArrowPatch((x, y_top), (x, y_bot), arrowstyle="-|>",
                                 mutation_scale=13, lw=1.4, color="#34495e"))


box(18.7, 1.0, "token ids   (B, L)     L = prot_max_len + 1 + mol_max_len", fc="#fdfefe", fs=8.3)
down(18.7, 17.9)
box(16.7, 1.0, "nn.Embedding(vocab = 71, d_model)")
down(16.7, 15.9)
box(14.7, 1.0, "+  sinusoidal PositionalEncoding")
down(14.7, 13.9)

# repeated transformer block
ax.add_patch(FancyBboxPatch((0.5, 4.4), 9.0, 9.0, boxstyle="round,pad=0.1",
                            linewidth=1.7, edgecolor="#2e4053", facecolor="#f4f7fb",
                            linestyle=(0, (6, 4))))
ax.text(4.7, 13.75, "×  num_layers", ha="center", va="bottom", fontsize=9.5,
        style="italic", color="#2e4053")
box(11.5, 1.5, "Multi-Head Self-Attention  (batch_first)\n"
               "+ key_padding_mask   + prefix-LM attn_mask", fs=7.8, w=8.0)
down(11.5, 10.7)
box(9.3, 1.0, "residual add   →   LayerNorm", w=8.0, fs=8.4)
down(9.3, 8.5)
box(6.8, 1.5, "Feed-Forward\nLinear(d_model → ff) · ReLU · Linear(ff → d_model)", w=8.0, fs=7.8)
down(6.8, 6.0)
box(4.8, 1.0, "residual add   →   LayerNorm", w=8.0, fs=8.4)

down(4.4, 3.6)
box(2.4, 1.0, "Linear(d_model → vocab = 71)")
down(2.4, 1.6)
box(0.3, 1.0, "logits  (B, L, 71)   →   CrossEntropy  /  sampling", fc="#fdfefe", fs=8.3)

# ============================================================================
# (3) PREFIX-LM ATTENTION MASK for the toy example
# ============================================================================
ax = fig.add_subplot(gs[1, 1])
toks = ["<cls>", "M", "A", "R", "C", "<DELIM>", "C", "C", "O"]
n, start = len(toks), 5
M = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i < start:
            M[i, j] = 0 if j < start else 1
        else:
            M[i, j] = 0 if j <= i else 1
ax.imshow(M, cmap="Greys", vmin=0, vmax=1.7)
ax.set_xticks(range(n)); ax.set_xticklabels(toks, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(n)); ax.set_yticklabels(toks, fontsize=8)
ax.set_xlabel("key  (attended-to position)", fontsize=9)
ax.set_ylabel("query position", fontsize=9)
ax.set_title("3 · Prefix-LM attention mask   (white = allowed, black = −inf)",
             fontsize=12, fontweight="bold", loc="left", pad=16)
ax.axhline(start - 0.5, color="#c0392b", lw=1.5)
ax.axvline(start - 0.5, color="#c0392b", lw=1.5)
ax.text(2.0, 2.0, "prefix ↔ prefix\nfull", ha="center", va="center", fontsize=8.5, color="#21618c")
ax.text(7.0, 1.5, "prefix → suffix\nblocked", ha="center", va="center", fontsize=8.5, color="white")
ax.text(2.0, 7.0, "suffix → prefix\nfull", ha="center", va="center", fontsize=8.5, color="#1e8449")
ax.text(7.2, 7.2, "suffix → suffix\ncausal", ha="center", va="center", fontsize=8.5, color="#1e8449")

# ============================================================================
# (4) TEACHER FORCING SHIFT + LABEL MASKING
# ============================================================================
ax = fig.add_subplot(gs[2, :])
ax.set_xlim(0, 20)
ax.set_ylim(0, 4.4)
ax.axis("off")
ax.set_title("4 · Teacher forcing  ·  next-token targets  ·  only suffix tokens are scored",
             fontsize=12, fontweight="bold", loc="left")

full = ["<cls>", "M", "A", "R", "C", "<DELIM>", "C", "C", "O", "<eos>"]
w, x0 = 1.35, 3.4


def row(y, items, title, colours):
    ax.text(x0 - 0.3, y + 0.45, title, ha="right", va="center", fontsize=9, fontweight="bold")
    for k, (label, colour) in enumerate(zip(items, colours)):
        ax.add_patch(FancyBboxPatch((x0 + k * w, y), w * 0.9, 0.9, boxstyle="round,pad=0.02",
                                    linewidth=1.0, edgecolor="#34495e", facecolor=colour))
        ax.text(x0 + k * w + w * 0.45, y + 0.45, label, ha="center", va="center", fontsize=8.5)


row(3.0, full[:-1], "input  x[:-1]",
    [C_SPECIAL, C_PROT, C_PROT, C_PROT, C_PROT, C_DELIM, C_MOL, C_MOL, C_MOL])
row(1.5, ["-100", "-100", "-100", "-100", "-100", "-100", "C", "O", "<eos>"], "labels  x[1:]",
    [C_PAD, C_PAD, C_PAD, C_PAD, C_PAD, C_PAD, C_MOL, C_MOL, C_EOS])

for k in range(9):
    ax.add_patch(FancyArrowPatch((x0 + k * w + w * 0.45, 2.95), (x0 + k * w + w * 0.45, 2.45),
                                 arrowstyle="-|>", mutation_scale=9, lw=0.9, color="#95a5a6"))
ax.add_patch(FancyBboxPatch((x0 + 6 * w - 0.08, 1.42), w * 3.0, 1.05, boxstyle="round,pad=0.02",
                            linewidth=1.8, edgecolor=C_LOSS, facecolor="none"))
ax.text(x0 + 7.5 * w, 0.85, "loss on these 3 positions only", ha="center", fontsize=8.5,
        color="#c0392b")
ax.text(x0 + 9 * w + 0.4, 1.95,
        "amino-acid, <pad> and <DELIM> targets → -100  (ignored by CrossEntropy)",
        ha="left", va="center", fontsize=8.5, color="#7f8c8d")

fig.suptitle("genAI  —  protein-conditioned molecule generator  (prefix-decoder Transformer)",
             fontsize=14, fontweight="bold", y=0.94)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=130, bbox_inches="tight")
print("wrote", os.path.abspath(OUT))
