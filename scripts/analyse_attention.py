"""How much does the molecule actually attend to the protein prefix?

The separability tests measure the prefix's effect on OUTPUTS. This measures it
inside the network, which answers a different and more mechanistic question: if
the prefix barely receives attention, or receives it in a way that ignores the
sequence's content, then "the prefix only controls molecular size" has a cause
rather than just a symptom.

The model is a prefix-LM: [<cls> protein residues] <DELIM> [SMILES tokens].
Prefix positions attend to each other bidirectionally; suffix (SMILES) positions
attend to the whole prefix plus causally to earlier suffix positions
(models/decoder_model.py, create_prefix_decoder_mask). So every SMILES token is
ALLOWED to look at every residue, and the question is whether it does.

Four readouts, per layer and per head:

  1. prefix mass      -- the fraction of a SMILES token's attention that lands on
                         the prefix at all. If this is near zero, the protein is
                         decoration.
  2. positional decay -- prefix mass as a function of position along the SMILES.
                         A molecule that consults the protein only for its first
                         few tokens would explain a size-only effect: early
                         tokens set the scaffold scale, the rest is language
                         modelling.
  3. where it lands   -- the distribution over prefix positions. Attention piled
                         onto <cls> or onto <DELIM> is a constant bias, not
                         sequence reading: it carries no per-residue information,
                         so every target looks alike. Reported as the share on
                         <cls>, the share on <DELIM>, and the normalised entropy
                         of the profile over residues (1.0 = perfectly uniform,
                         0.0 = one position takes everything).
  4. content test     -- the same molecules scored against a SHUFFLED prefix
                         (same residues, order destroyed). If attention is
                         unchanged, the model is responding to amino-acid
                         composition at most, not to sequence. This is the
                         sharpest available check and needs no extra training.

Teacher-forced forward passes on each checkpoint's OWN generations for that
target, with dropout disabled, so the attention being measured is the attention
that produced those molecules.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightning.fabric import Fabric

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from models.decoder_model import MultiLayerTransformerDecoder
from utils.tokenizer import Tokenizer

BASE = os.environ.get("PREFIX_GEN_OUT", os.path.join(ROOT, "tests", "prefix_generation"))
OUTDIR = os.path.join(BASE, "analysis", "attention")

TARGETS = ["CDK2", "very_low_6EI4", "low_6LHR", "medium_2JBH", "high_8B4T"]
SHORT = {"CDK2": "CDK2", "very_low_6EI4": "6EI4", "low_6LHR": "6LHR",
         "medium_2JBH": "2JBH", "high_8B4T": "8B4T"}
CKPT_WEIGHTS = {
    "partB_val_30M": ("partB_val", "weights_dm256_nh4_ff2560_nl19_params_30M.pth"),
    "partB_sched_30M": ("partB_sched", "weights_dm256_nh4_ff2560_nl19_params_30M.pth"),
    "partB_val_40M": ("partB_val", "weights_dm256_nh4_ff4096_nl17_params_40M.pth"),
    "partB_sched_40M": ("partB_sched", "weights_dm256_nh4_ff4096_nl17_params_40M.pth"),
}

SEED = 1234
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#8a8984"]
NEUTRALC = "#8a8984"
# Sequential = ONE hue, light -> dark, built from the categorical blue.
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#ffffff", "#dce8f8", "#9cc0ee", "#2a78d6", "#14406f"])


class CaptureAttention(nn.Module):
    """Wraps a block's nn.MultiheadAttention to keep the per-head weights.

    The model throws them away (`attn_output, _ = self.self_attention(...)`) and
    nn.MultiheadAttention averages over heads unless told otherwise, so both
    need_weights and average_attn_weights have to be set here. This forces the
    slow attention path, which is fine for analysis and wrong for training."""

    def __init__(self, mha):
        super().__init__()
        self.mha = mha
        self.weights = None

    def forward(self, q, k, v, key_padding_mask=None, attn_mask=None):
        out, w = self.mha(q, k, v, key_padding_mask=key_padding_mask,
                          attn_mask=attn_mask, need_weights=True,
                          average_attn_weights=False)
        self.weights = w.detach()          # (batch, heads, seq, seq)
        return out, w


def load_model(ckpt, fabric):
    regime, fname = CKPT_WEIGHTS[ckpt]
    path = os.path.join(ROOT, "weights", regime, fname)
    if not os.path.exists(path):
        raise SystemExit(f"missing weights: {path}")
    import re
    d_model, num_heads, ff, num_layers = (
        int(g) for g in re.search(r"dm(\d+)_nh(\d+)_ff(\d+)_nl(\d+)", fname).groups())
    tok = Tokenizer()
    model = MultiLayerTransformerDecoder(tok.vocab_size, d_model, num_heads, ff, 0.25, num_layers)
    fabric.load(path, {"model": model})
    model.eval()                            # dropout is 0.25 -- must be off
    for blk in model.transformer_blocks:
        blk.self_attention = CaptureAttention(blk.self_attention)
    return model, tok, num_layers, num_heads


def read_fasta(path):
    seqs, name = {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            name = line[1:].split()[0]
            seqs[name] = ""
        elif name:
            seqs[name] += line
    return seqs


def molecules(ckpt, target, n, rng):
    path = os.path.join(BASE, ckpt, target, "molecules_unique.smi")
    smis = [l.split()[0] for l in open(path) if l.strip()]
    idx = rng.choice(len(smis), min(n, len(smis)), replace=False)
    return [smis[i] for i in idx]


@torch.no_grad()
def attention_stats(model, tok, seq, smis, fabric, batch_size, n_layers, n_heads,
                    max_suffix=60, n_profile=None):
    """Accumulate the four readouts over a set of molecules for one prefix.

    Returns
      mass     (layers, heads)            mean prefix mass per suffix token
      decay    (layers, suffix positions) prefix mass vs position along SMILES
      profile  (layers, prefix positions) attention each prefix position receives
      delim    (layers, heads)            mass on the <DELIM> token alone
    """
    mass = np.zeros((n_layers, n_heads))
    delim = np.zeros((n_layers, n_heads))
    decay = np.zeros((n_layers, max_suffix))
    decay_n = np.zeros(max_suffix)
    profile = None
    n_seen = 0

    for start in range(0, len(smis), batch_size):
        chunk = smis[start:start + batch_size]
        enc = tok([seq] * len(chunk), chunk)
        ids, pad = enc["input_ids"], enc["attention_mask"]

        model(ids, pad, tok.delim_token_id, fabric)

        d = (ids[0] == tok.delim_token_id).nonzero(as_tuple=True)[0][0].item()
        is_pad = pad.to(dtype=torch.bool).cpu().numpy()          # (B, L) True == pad
        L = ids.shape[1]
        # Suffix queries: after <DELIM>, real tokens only. The <DELIM> row itself
        # is excluded as a query -- it has no molecule context yet.
        if profile is None:
            profile = np.zeros((n_layers, d + 1))

        for li, blk in enumerate(model.transformer_blocks):
            w = blk.self_attention.weights.float().cpu().numpy()  # (B, H, L, L)
            for bi in range(w.shape[0]):
                q = np.flatnonzero(~is_pad[bi])
                q = q[q > d]
                if q.size == 0:
                    continue
                a = w[bi][:, q, :]                                # (H, Q, L)
                pm = a[:, :, :d].sum(axis=2)                      # prefix (no DELIM)
                mass[li] += pm.mean(axis=1)
                delim[li] += a[:, :, d].mean(axis=1)
                head_mean = pm.mean(axis=0)                       # over heads -> (Q,)
                k = min(head_mean.size, max_suffix)
                decay[li, :k] += head_mean[:k]
                if li == 0:
                    decay_n[:k] += 1
                profile[li] += a[:, :, :d + 1].sum(axis=(0, 1)) / (a.shape[0] * q.size)
        n_seen += len(chunk)

    # mass/delim/profile accumulate once per molecule; main() divides by n_seen.
    return mass, delim, decay, decay_n, profile, n_seen



def probe(fabric, ckpt):
    """Is the position-64 cliff positional, or a property of the sequence?

    Same model, prefixes that vary in length, in which part of the protein they
    come from, and in content. If the cliff stays at the same ABSOLUTE index in
    all of them, the model is reading a fixed-size window of whatever it is
    given, and the rest of the protein is nearly invisible to it."""
    model, tok, nl, nh = load_model(ckpt, fabric)
    model = fabric.to_device(model)
    seqs = read_fasta(os.path.join(BASE, "targets.fasta"))
    cdk2 = seqs["CDK2"]
    mol = [open(os.path.join(BASE, ckpt, "CDK2", "molecules_unique.smi")).readline().split()[0]]
    rng = np.random.default_rng(SEED)

    cases = {
        "CDK2 real": cdk2,
        "CDK2 first 150": cdk2[:150],
        "CDK2 last 150": cdk2[-150:],
        "poly-A": "A" * len(cdk2),
        "random residues": "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), len(cdk2))),
    }
    blocks = [(0, 32), (32, 64), (64, 96), (96, 128), (128, 600)]
    print(f"\n=== positional probe, {ckpt} ===")
    print("attention per prefix position, x uniform (1.00 = fair share)")
    print(f"{'prefix':18}{'len':>5}" + "".join(f"{f'{a}-{b}':>10}" for a, b in blocks))
    rows = []
    for name, seq in cases.items():
        _, _, _, _, prof, n = attention_stats(model, tok, seq, mol, fabric, 1, nl, nh)
        prof = prof / n
        r = prof[:, :-1]
        r = r / r.sum(axis=1, keepdims=True)
        pm = r.mean(axis=0)
        L = pm.size
        line, rec = f"{name:18}{L:5}", dict(prefix=name, prefix_len=L)
        for a, b in blocks:
            v = pm[a:min(b, L)].mean() * L if a < L else np.nan
            line += f"{v:10.2f}" if not np.isnan(v) else f"{'--':>10}"
            rec[f"x_uniform_{a}_{b}"] = None if np.isnan(v) else round(float(v), 3)
        print(line)
        rows.append(rec)
    pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, "attention_positional_probe.csv"), index=False)
    print(f"wrote {os.path.join(OUTDIR, 'attention_positional_probe.csv')}")


def example(fabric, ckpt, target, mol_rank, layer_show):
    """One prefix, one molecule: the attention matrix itself, not a summary.

    The aggregate plots average over 64 molecules, five targets, every head and
    every layer -- which is what makes them trustworthy and also what makes
    them abstract. This renders a single forward pass, so every cell on the
    page belongs to one named SMILES token attending to one named residue.

    Colour answers WHERE a token looks (each row normalised to its own protein
    attention) and the marginal bar answers HOW MUCH it looks at the protein at
    all. Splitting the two keeps one colour scale from doing two jobs, which is
    what made the window at position 64 invisible in the summary statistics.
    """
    model, tok, nl, nh = load_model(ckpt, fabric)
    model = fabric.to_device(model)
    seqs = read_fasta(os.path.join(BASE, "targets.fasta"))

    hit = [t for t in TARGETS if t == target or SHORT[t] == target]
    if not hit:
        raise SystemExit(f"unknown target {target!r}; choose from "
                         + ", ".join(SHORT.values()))
    tname = hit[0]
    seq = seqs[tname]
    smis = [l.split()[0] for l in
            open(os.path.join(BASE, ckpt, tname, "molecules_unique.smi")) if l.strip()]
    if mol_rank >= len(smis):
        raise SystemExit(f"only {len(smis)} molecules for {tname}")
    smi = smis[mol_rank]

    enc = tok([seq], [smi])
    ids, pad = enc["input_ids"], enc["attention_mask"]
    with torch.no_grad():
        model(ids, pad, tok.delim_token_id, fabric)
    W = np.stack([blk.self_attention.weights[0].float().cpu().numpy()
                  for blk in model.transformer_blocks])      # (layers, heads, L, L)

    toks = [tok.id2token[int(i)] for i in ids[0]]
    d = toks.index(tok.delim_token)
    is_pad = pad[0].to(dtype=torch.bool).cpu().numpy()        # True == <pad>
    q = np.flatnonzero(~is_pad)
    q = q[q > d]                                              # SMILES queries only

    A = W[:, :, q, :d + 1]                                    # (layers, heads, Q, d+1)
    Ah = A.mean(axis=1)                                       # over heads
    Am = Ah.mean(axis=0)                                      # over layers
    prot = Am[:, :d]                                          # <cls> + residues
    delim = Am[:, d]
    row_mass = prot.sum(axis=1)
    row_norm = prot / np.clip(row_mass[:, None], 1e-12, None)

    res = ["<cls>"] + list(seq)                               # d labels
    ylab = [toks[i] for i in q]
    Q = len(q)
    tag = f"{ckpt.replace('partB_', '')}, {SHORT[tname]}"

    print(f"\n=== one example: {ckpt}  {SHORT[tname]}  molecule #{mol_rank} ===")
    print(f"  SMILES   {smi}")
    print(f"  prefix   {len(seq)} residues ({d} prefix tokens incl. <cls>)")
    print(f"  suffix   {Q} SMILES query tokens")
    print(f"  prefix mass: first token {row_mass[0]*100:.1f}%  "
          f"last token {row_mass[-1]*100:.1f}%  mean {row_mass.mean()*100:.1f}%")
    print(f"  <DELIM> mass: mean {delim.mean()*100:.2f}%")
    inw = row_norm[:, :64].sum(axis=1).mean() * 100
    print(f"  share of protein attention inside positions 0-63: {inw:.1f}% "
          f"({64/d*100:.1f}% of the prefix)")

    # ---------------------------------------------------------------- figure 1
    fig = plt.figure(figsize=(14.5, 10.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 0.17, 0.022],
                          height_ratios=[1, 0.82], hspace=0.30, wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1], sharey=ax)
    axc = fig.add_subplot(gs[0, 2])
    axz = fig.add_subplot(gs[1, :2])
    axzc = fig.add_subplot(gs[1, 2])

    vmax = np.percentile(row_norm, 99.5)
    im = ax.imshow(row_norm * 100, aspect="auto", cmap=SEQ, vmin=0, vmax=vmax * 100,
                   interpolation="nearest")
    ax.set_yticks(range(Q))
    ax.set_yticklabels(ylab, fontsize=6.5, color=INK2)
    ticks = [0] + [t for t in (64, 100, 150, 200, 250, 300, 350, 400) if t < d]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], fontsize=8.5, color=INK2)
    ax.axvline(63.5, color=PAL[1], lw=1.6, ls="--")
    ax.annotate("attention window ends (position 64)", xy=(64, Q * 0.06),
                xytext=(86, Q * 0.06), fontsize=9, color=PAL[1], va="center",
                arrowprops=dict(arrowstyle="-|>", color=PAL[1], lw=1.2))
    ax.set_xlabel("prefix position (0 = <cls>, then residue index)", fontsize=10, color=INK2)
    ax.set_ylabel("SMILES token (generation order)", fontsize=10, color=INK2)
    ax.set_title(f"Where each SMILES token looks in the protein\n{tag}, molecule #{mol_rank}"
                 "  -- each row normalised to its own protein attention",
                 fontsize=11.5, color=INK, loc="left")
    cb = plt.colorbar(im, cax=axc)
    cb.set_label("share of the token's protein attention (%)", fontsize=8.5, color=INK2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7.5, colors=INK2)

    axb.barh(np.arange(Q), row_mass * 100, 0.82, color=PAL[0], zorder=3)
    axb.set_xlabel("on protein (%)", fontsize=9, color=INK2)
    axb.set_title("how much\nit looks", fontsize=10, color=INK, loc="left")
    axb.tick_params(labelleft=False, labelsize=8, colors=INK2)
    axb.grid(axis="x", color=GRID, lw=1, zorder=1)
    axb.set_axisbelow(True)
    # first and last rows carry the story -- label those two, not all Q
    for i in (0, Q - 1):
        axb.annotate(f"{row_mass[i]*100:.0f}%", xy=(row_mass[i] * 100, i),
                     xytext=(3, 0), textcoords="offset points", fontsize=8,
                     color=INK2, va="center")

    nz = min(64, d)
    imz = axz.imshow(row_norm[:, :nz] * 100, aspect="auto", cmap=SEQ, vmin=0,
                     vmax=vmax * 100, interpolation="nearest")
    axz.set_xticks(range(nz))
    axz.set_xticklabels(res[:nz], fontsize=6.5, color=INK2)
    axz.set_yticks(range(Q))
    axz.set_yticklabels(ylab, fontsize=6.5, color=INK2)
    axz.set_xlabel("residue inside the window that the model actually reads",
                   fontsize=10, color=INK2)
    axz.set_ylabel("SMILES token", fontsize=10, color=INK2)
    axz.set_title(f"Zoom on prefix positions 0-{nz-1}\n{tag} -- "
                  f"{inw:.0f}% of all protein attention lands in this strip",
                  fontsize=11.5, color=INK, loc="left")
    cbz = plt.colorbar(imz, cax=axzc)
    cbz.set_label("share (%)", fontsize=8.5, color=INK2)
    cbz.outline.set_visible(False)
    cbz.ax.tick_params(labelsize=7.5, colors=INK2)

    for a in (ax, axb, axz):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(GRID)

    p1 = os.path.join(OUTDIR, f"attention_example_{ckpt}_{SHORT[tname]}.png")
    fig.savefig(p1, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"wrote {p1}")

    # ---------------------------------------------------------------- figure 2
    # The whole sequence attending to itself, one layer: the prefix-LM mask made
    # visible. Split into two panels by query type -- the molecule is 40 rows
    # against the protein's 300, so a single square map hides the block that
    # matters. Log colour because the causal diagonal and the protein block are
    # three orders of magnitude apart; both panels share one scale.
    keep = np.flatnonzero(~is_pad)
    F = W[layer_show].mean(axis=0)[np.ix_(keep, keep)]
    npre = int((keep < d).sum())
    lo = max(float(F[F > 0].min()), 1e-5)
    norm = LogNorm(vmin=lo, vmax=float(F.max()))

    fig2 = plt.figure(figsize=(10.2, 8.2))
    gs2 = fig2.add_gridspec(2, 2, height_ratios=[1, 0.55], width_ratios=[1, 0.025],
                            hspace=0.12, wspace=0.03)
    axp = fig2.add_subplot(gs2[0, 0])
    axm = fig2.add_subplot(gs2[1, 0], sharex=axp)
    axcb = fig2.add_subplot(gs2[:, 1])

    for a, rows, ttl in (
            (axp, slice(0, npre), "protein queries -- the prefix attends to itself, "
                                  "bidirectionally"),
            (axm, slice(npre, None), "molecule queries -- all of the protein is "
                                     "allowed, plus causal within the molecule")):
        img = a.imshow(np.clip(F[rows], lo, None), cmap=SEQ, norm=norm,
                       aspect="auto", interpolation="nearest",
                       extent=(-0.5, len(keep) - 0.5,
                               (rows.stop if rows.stop else len(keep)) - 0.5,
                               rows.start - 0.5))
        a.axvline(d - 0.5, color=PAL[1], lw=1.2)
        a.axvline(63.5, color=PAL[2], lw=1.3, ls="--")
        a.set_title(ttl, fontsize=10, color=INK, loc="left", pad=4)
        a.set_ylabel("query position", fontsize=9.5, color=INK2)
        for s_ in ("top", "right", "left", "bottom"):
            a.spines[s_].set_visible(False)
    axp.tick_params(labelbottom=False, labelsize=8, colors=INK2)

    ticks = [0, 64] + [t for t in (100, 200, 300, 400) if t < d] + [d]
    axm.set_xticks(ticks)
    axm.set_xticklabels([("<DELIM>" if t == d else str(t)) for t in ticks],
                        fontsize=8.5, color=INK2)
    for lbl, t in zip(axm.get_xticklabels(), ticks):
        if t == 64:
            lbl.set_color(PAL[2])
        elif t == d:
            lbl.set_color(PAL[1])
    axm.set_xlabel("key position (what is attended to)  --  "
                   "dashed line = the position-64 window", fontsize=9.5, color=INK2)
    axm.tick_params(labelsize=8, colors=INK2)

    fig2.suptitle(f"The full prefix-LM attention map, layer {layer_show}\n{tag}, "
                  f"molecule #{mol_rank} -- mean over {nh} heads, log colour",
                  fontsize=11.5, color=INK, x=0.065, ha="left", y=0.995)
    cb2 = fig2.colorbar(img, cax=axcb)
    cb2.set_label("attention weight", fontsize=9, color=INK2)
    cb2.outline.set_visible(False)
    cb2.ax.tick_params(labelsize=8, colors=INK2)
    p2 = os.path.join(OUTDIR, f"attention_example_blocks_{ckpt}_{SHORT[tname]}.png")
    fig2.savefig(p2, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"wrote {p2}")

    # the per-row numbers, so the figure can be re-read without re-running
    pd.DataFrame(dict(
        position=np.arange(Q), token=ylab,
        prefix_mass=np.round(row_mass, 5), delim_mass=np.round(delim, 5),
        share_window_0_63=np.round(row_norm[:, :64].sum(axis=1), 5),
        share_cls=np.round(row_norm[:, 0], 5),
    )).to_csv(os.path.join(OUTDIR,
              f"attention_example_{ckpt}_{SHORT[tname]}.csv"), index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoints", nargs="+", default=list(CKPT_WEIGHTS))
    ap.add_argument("--n", type=int, default=64, help="molecules per target")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--probe", action="store_true",
                    help="only run the positional probe (cheap, CPU is fine)")
    ap.add_argument("--example", metavar="TARGET",
                    help="only plot the attention of ONE prefix/molecule pair, "
                         "e.g. --example CDK2 (cheap, CPU is fine)")
    ap.add_argument("--mol", type=int, default=0,
                    help="which molecule of that target to use (--example)")
    ap.add_argument("--layer", type=int, default=0,
                    help="layer for the full prefix-LM map (--example)")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    fabric = Fabric(accelerator=args.device, devices=1)
    fabric.launch()

    if args.probe:
        probe(fabric, args.checkpoints[0])
        return

    if args.example:
        example(fabric, args.checkpoints[0], args.example, args.mol, args.layer)
        return

    seqs = read_fasta(os.path.join(BASE, "targets.fasta"))
    rows, decay_store, profile_store, mass_store = [], {}, {}, {}

    for ckpt in args.checkpoints:
        model, tok, n_layers, n_heads = load_model(ckpt, fabric)
        model = fabric.to_device(model)
        print(f"\n=== {ckpt}  ({n_layers} layers x {n_heads} heads) ===")

        for target in TARGETS:
            seq = seqs[target]
            smis = molecules(ckpt, target, args.n, rng)
            for variant in ("real", "shuffled"):
                s = seq if variant == "real" else "".join(
                    rng.permutation(list(seq)))
                mass, delim, decay, decay_n, profile, n = attention_stats(
                    model, tok, s, smis, fabric, args.batch_size, n_layers, n_heads)
                mass /= n
                delim /= n
                with np.errstate(invalid="ignore", divide="ignore"):
                    decay = decay / np.where(decay_n > 0, decay_n, np.nan)
                profile = profile / n

                res = np.array([p[:-1] for p in profile])       # drop DELIM column
                res_norm = res / np.clip(res.sum(axis=1, keepdims=True), 1e-12, None)
                ent = -(res_norm * np.log(np.clip(res_norm, 1e-12, None))).sum(axis=1)
                ent /= np.log(res.shape[1])
                cls_share = res_norm[:, 0]

                rows.append(dict(
                    checkpoint=ckpt, target=SHORT[target], prefix=variant,
                    n_molecules=n, prefix_len=res.shape[1],
                    prefix_mass=round(float(mass.mean()), 4),
                    prefix_mass_max_head=round(float(mass.max()), 4),
                    delim_mass=round(float(delim.mean()), 4),
                    cls_share_of_prefix=round(float(cls_share.mean()), 4),
                    profile_entropy=round(float(ent.mean()), 4),
                ))
                key = (ckpt, SHORT[target], variant)
                decay_store[key] = decay
                profile_store[key] = res_norm
                mass_store[key] = mass
                print(f"  {SHORT[target]:5} {variant:9} prefix mass "
                      f"{mass.mean():.3f} (max head {mass.max():.3f})   "
                      f"DELIM {delim.mean():.3f}   <cls> share of prefix "
                      f"{cls_share.mean():.3f}   entropy {ent.mean():.3f}")

        # free the captured weights before the next checkpoint
        for blk in model.transformer_blocks:
            blk.self_attention.weights = None
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUTDIR, "attention_summary.csv"), index=False)
    np.savez_compressed(os.path.join(OUTDIR, "attention_arrays.npz"),
                        **{"decay__" + "__".join(k): v for k, v in decay_store.items()},
                        **{"profile__" + "__".join(k): v for k, v in profile_store.items()},
                        **{"mass__" + "__".join(k): v for k, v in mass_store.items()})
    print(f"\nwrote {os.path.join(OUTDIR, 'attention_summary.csv')}")

    summarise(R)
    plot(R, decay_store, profile_store, mass_store, args.checkpoints)


def summarise(R):
    print("\n=== prefix mass: fraction of each SMILES token's attention on the protein ===")
    real = R[R.prefix == "real"].pivot(index="checkpoint", columns="target", values="prefix_mass")
    print(real.round(3).to_string())

    print("\n=== real vs shuffled prefix (is the model reading the SEQUENCE?) ===")
    for ckpt in R.checkpoint.unique():
        d = R[R.checkpoint == ckpt]
        a = d[d.prefix == "real"].set_index("target").prefix_mass
        b = d[d.prefix == "shuffled"].set_index("target").prefix_mass
        diff = (a - b).abs().mean()
        rel = diff / max(a.mean(), 1e-9) * 100
        verdict = ("content-sensitive" if rel >= 5 else
                   "INSENSITIVE to residue order -- composition at most")
        print(f"  {ckpt:18} mean |real - shuffled| = {diff:.4f} "
              f"({rel:.1f}% of prefix mass)   {verdict}")

    print("\n=== how concentrated is prefix attention? ===")
    g = R[R.prefix == "real"].groupby("checkpoint").agg(
        cls_share=("cls_share_of_prefix", "mean"), entropy=("profile_entropy", "mean"),
        delim=("delim_mass", "mean"))
    print(g.round(3).to_string())
    print("\n  <cls> share near 1.0 means the prefix is read as a single bias token.")
    print("  entropy near 1.0 means attention is spread evenly over residues.")


def plot(R, decay_store, profile_store, mass_store, checkpoints):
    fig, axes2d = plt.subplots(2, 2, figsize=(13.5, 9.4))
    axes = [axes2d[0, 0], axes2d[0, 1], axes2d[1, 0]]
    axheat = axes2d[1, 1]

    # 1. prefix mass per checkpoint x target
    ax = axes[0]
    real = R[R.prefix == "real"].pivot(index="checkpoint", columns="target", values="prefix_mass")
    real = real.reindex([c for c in checkpoints if c in real.index])
    x = np.arange(len(real))
    w = 0.16
    for i, t in enumerate(real.columns):
        ax.bar(x + (i - 2) * w, real[t] * 100, w * 0.9, label=t, color=PAL[i],
               edgecolor="white", linewidth=1.5, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("partB_", "").replace("_", " ") for c in real.index],
                       fontsize=9, color=INK2)
    ax.set_ylabel("attention on prefix (%)", fontsize=10, color=INK2)
    ax.set_title("How much of a SMILES token's attention\nreaches the protein",
                 fontsize=11.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.5, ncol=2)

    # 2. decay along the molecule
    ax = axes[1]
    ck = checkpoints[0]
    for i, t in enumerate(SHORT.values()):
        k = (ck, t, "real")
        if k not in decay_store:
            continue
        d = np.nanmean(decay_store[k], axis=0) * 100
        ax.plot(np.arange(d.size), d, color=PAL[i], lw=2, label=t)
    ax.set_xlabel("position along the SMILES", fontsize=10, color=INK2)
    ax.set_ylabel("attention on prefix (%)", fontsize=10, color=INK2)
    ax.set_title(f"Does the prefix fade as the molecule grows?\n{ck.replace('partB_','')}, "
                 "mean over layers and heads", fontsize=11.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.5, ncol=2)

    # 3. where in the prefix attention lands
    ax = axes[2]
    k = (ck, "CDK2", "real")
    ks = (ck, "CDK2", "shuffled")
    if k in profile_store:
        p = profile_store[k].mean(axis=0) * 100
        ax.plot(np.arange(p.size), p, color=PAL[0], lw=1.6, label="real prefix")
        if ks in profile_store:
            q = profile_store[ks].mean(axis=0) * 100
            ax.plot(np.arange(q.size), q, color=NEUTRALC, lw=1.4, ls="--",
                    label="shuffled prefix")
        ax.set_yscale("log")
    ax.set_xlabel("prefix position (0 = <cls>)", fontsize=10, color=INK2)
    ax.set_ylabel("share of prefix attention (%, log)", fontsize=10, color=INK2)
    ax.set_title(f"Where in the protein does it look?\n{ck.replace('partB_','')}, CDK2",
                 fontsize=11.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.5)

    # 4. the layer x head map the scalar means average away
    ck = checkpoints[0]
    key = (ck, "CDK2", "real")
    if key in mass_store:
        M = mass_store[key] * 100
        im = axheat.imshow(M, aspect="auto", cmap="Blues", vmin=0,
                           vmax=max(100 * 0.9, M.max()))
        axheat.set_xlabel("head", fontsize=10, color=INK2)
        axheat.set_ylabel("layer", fontsize=10, color=INK2)
        axheat.set_xticks(range(M.shape[1]))
        axheat.set_title(f"Prefix mass per layer and head\n{ck.replace('partB_','')}, CDK2 "
                         "(%, white = ignores the protein)",
                         fontsize=11.5, color=INK, loc="left")
        cb = plt.colorbar(im, ax=axheat, fraction=0.046, pad=0.02)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=8, colors=INK2)
        for s_ in ("top", "right", "left", "bottom"):
            axheat.spines[s_].set_visible(False)

    for ax in axes:
        ax.grid(color=GRID, lw=1, zorder=1)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)

    fig.tight_layout()
    p = os.path.join(OUTDIR, "attention.png")
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
