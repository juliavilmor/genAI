"""Sequence identity between a test split and the training set, for any band.

Identity is computed the same way as data_preparation/get_SMP_splits.py built
the splits in the first place -- Bio.Align.PairwiseAligner, global mode,
identity = matches / max(len(seq1), len(seq2)), and a sequence's identity "to
training" is the maximum over the whole training set. Bands are half-open,
[min, max), so a sequence at exactly 90 belongs to the 90+ bucket, not to
60-90.

Two stages (this was verify_test_60_90_identity.py and
add_test_60_90_sequence.py, both hard-coded to the 60-90 band):

  verify  recompute each sequence's identity to training and report whether it
          is genuinely in the band. Worth doing whenever training changes:
          removing a sequence from training moves every split sequence's
          "identity to training" from whatever it was to its next-nearest
          remaining neighbour, which can push it out of band.

  find    draw candidates from the SMPBind pool and keep the first whose
          identity to training is in the band and that is not already used in
          training or in any split -- i.e. one new row for a split that is
          short. Measuring against training (not against the pool) and
          excluding already-used sequences is what makes this different from
          get_SMP_splits.py's range sampler.

All paths are relative to the project root, so run this from there, not from
inside data_preparation/ (unlike get_SMP_splits.py, whose paths start "../").

Both stages are single-threaded and slow: ~15-20 min for 99 queries against
~16.3k training sequences. Run them as a CPU job, not on a login node (see
LAB_NOTEBOOK B.5 re the login-node process reaper):

    python data_preparation/split_identity.py verify --split data/splits/test_60_90.csv --min 60 --max 90
    python data_preparation/split_identity.py find   --split data/splits/test_60_90.csv --min 60 --max 90 --append
"""
import argparse
import os
import sys
import time

import pandas as pd
from Bio.Align import PairwiseAligner

TRAIN_PATH = "data/splits/training_split.csv"
POOL_PATH = "data/data_SMPBind_downsampled_50.csv"
SPLIT_PATHS = [
    "data/splits/test_90.csv",
    "data/splits/test_60_90.csv",
    "data/splits/test_40_60.csv",
    "data/splits/test_0_40.csv",
]

aligner = PairwiseAligner()
aligner.mode = "global"


def identity(seq1, seq2):
    a = aligner.align(seq1, seq2)[0]
    matches = sum(1 for x, y in zip(a[0], a[1]) if x == y)
    return matches / max(len(seq1), len(seq2)) * 100


def reference_sequences(path):
    df = pd.read_csv(path, usecols=["Sequence"])
    seqs = df["Sequence"].drop_duplicates().tolist()
    print(f"reference pool ({path}): {len(seqs)} unique sequences")
    return seqs


def max_identity_to(seq, reference):
    return max(identity(seq, s) for s in reference)


def cmd_verify(args):
    reference = reference_sequences(args.reference)
    split = pd.read_csv(args.split)
    print(f"{args.split}: {len(split)} rows")
    print(f"band: [{args.min}, {args.max})\n", flush=True)

    rows = []
    t0 = time.time()
    for i, seq in enumerate(split["Sequence"]):
        best = max_identity_to(seq, reference)
        rows.append({"Uniprot": split["Uniprot"].iloc[i], "Sequence": seq,
                     "max_identity_to_reference": best,
                     "in_band": args.min <= best < args.max})
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(split)}  elapsed {elapsed:.0f}s  "
                  f"est. total {elapsed/(i+1)*len(split):.0f}s", flush=True)

    out = pd.DataFrame(rows)
    outpath = args.out or os.path.join(
        os.path.dirname(args.split),
        os.path.splitext(os.path.basename(args.split))[0] + "_identity_check.csv")
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    out.to_csv(outpath, index=False)
    print(f"\nwrote {outpath}")

    n_bad = int((~out["in_band"]).sum())
    print(f"{len(out) - n_bad}/{len(out)} genuinely in [{args.min}, {args.max}) identity")
    if n_bad:
        print(f"{n_bad} out of band:")
        print(out[~out["in_band"]][["Uniprot", "max_identity_to_reference"]]
              .to_string(index=False))
    return 0 if n_bad == 0 else 1


def cmd_find(args):
    reference = reference_sequences(args.reference)

    # Anything already in training or in any split is off limits, so a "new"
    # sequence is genuinely new to the whole experiment.
    used = set(pd.read_csv(args.reference, usecols=["Sequence"])["Sequence"])
    for path in args.exclude:
        if not os.path.exists(path):
            print(f"  (skipping {path}: not found)")
            continue
        df = pd.read_csv(path)
        used |= set(df["Sequence"])
        print(f"{path}: {len(df)} rows")

    pool = pd.read_csv(args.pool).drop_duplicates("Sequence", keep="first")
    candidates = pool[~pool["Sequence"].isin(used)].reset_index(drop=True)
    print(f"candidate pool ({args.pool}, deduped, unused): {len(candidates)}")
    candidates = candidates.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    print(f"band: [{args.min}, {args.max})  seed={args.seed}\n", flush=True)

    found = []
    t0 = time.time()
    for i, row in candidates.iterrows():
        if i >= args.max_attempts:
            print(f"gave up after {args.max_attempts} attempts without enough hits")
            break
        best = max_identity_to(row["Sequence"], reference)
        print(f"  attempt {i+1}: {row['Uniprot']}  identity={best:.2f}%  "
              f"(elapsed {time.time() - t0:.0f}s)", flush=True)
        if args.min <= best < args.max:
            found.append(row[["Uniprot", "Compound_ID", "SMILES", "Sequence"]])
            print(f"  ACCEPTED: {row['Uniprot']} / {row['Compound_ID']}  "
                  f"identity={best:.2f}%")
            if len(found) == args.n:
                break

    if not found:
        print("\nno candidate in band")
        return 1

    new_rows = pd.DataFrame(found)
    print(f"\n{len(found)} candidate(s) in [{args.min}, {args.max}):")
    print(new_rows[["Uniprot", "Compound_ID"]].to_string(index=False))

    if args.append:
        split = pd.read_csv(args.split)
        updated = pd.concat([split, new_rows], ignore_index=True)
        updated.to_csv(args.split, index=False)
        print(f"appended to {args.split}, now {len(updated)} rows "
              f"(was {len(split)})")
    else:
        print("not appended (pass --append to add them to the split)")
    return 0 if len(found) == args.n else 1


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    def common(p):
        p.add_argument("--split", required=True, help="the split CSV to work on")
        p.add_argument("--min", type=float, required=True,
                       help="lower edge of the identity band, inclusive")
        p.add_argument("--max", type=float, required=True,
                       help="upper edge of the identity band, exclusive")
        p.add_argument("--reference", default=TRAIN_PATH,
                       help=f"what identity is measured against (default {TRAIN_PATH})")

    p = sub.add_parser("verify", help="recompute identity for every sequence in a split")
    common(p)
    p.add_argument("--out", default=None,
                   help="output CSV (default: <split>_identity_check.csv next to the split)")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("find", help="find unused sequences whose identity is in the band")
    common(p)
    p.add_argument("--pool", default=POOL_PATH, help=f"candidate pool (default {POOL_PATH})")
    p.add_argument("--exclude", nargs="*", default=SPLIT_PATHS,
                   help="CSVs whose sequences are already in use (default: the 4 splits)")
    p.add_argument("--n", type=int, default=1, help="how many to accept")
    p.add_argument("--max-attempts", type=int, default=300)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--append", action="store_true",
                   help="append the accepted rows to --split (writes the file)")
    p.set_defaults(func=cmd_find)

    args = parser.parse_args()
    if args.min >= args.max:
        parser.error(f"--min ({args.min}) must be below --max ({args.max})")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
