"""Results analysis for the report: per-dataset and per-attribute tables + prior baseline.
python analyze.py --ckpt runs\cnxs\best.pt --train_ann ...gt.csv --val_ann ...gt.csv --img_root data --out results
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import upar_par as U


def per_attribute(prob, y, mask, thr, names):
    pred = (prob >= thr[None]).astype(np.float32)
    rows = []
    for a, n in enumerate(names):
        m = mask[:, a] > 0
        yy, pp = y[m, a], pred[m, a]
        tp = float(((pp == 1) & (yy == 1)).sum()); fn = float(((pp == 0) & (yy == 1)).sum())
        tn = float(((pp == 0) & (yy == 0)).sum()); fp = float(((pp == 1) & (yy == 0)).sum())
        tpr = tp / (tp + fn) if tp + fn else np.nan
        tnr = tn / (tn + fp) if tn + fp else np.nan
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan
        rows.append(dict(attribute=n, positive_rate=round(float(yy.mean()) * 100, 2), TPR=round(tpr * 100, 1),
                         TNR=round(tnr * 100, 1), mA=round(np.nanmean([tpr, tnr]) * 100, 1), F1=round(f1 * 100, 1)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--train_ann", required=True)
    ap.add_argument("--val_ann", required=True); ap.add_argument("--img_root", default="data")
    ap.add_argument("--out", default="results"); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--bs", type=int, default=128)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(a.ckpt, map_location=dev)
    names = ck["attributes"]; A = len(names)
    thr = np.array(ck.get("thr", [0.5] * A), dtype=np.float32)
    tr = U.load_table(a.train_ann); va = U.load_table(a.val_ann)
    pc, lc = U.split_table(va)
    assert lc == names, "attribute order differs between checkpoint and annotations"
    ytr, mtr = U.clean_labels(tr[lc].values); y, m = U.clean_labels(va[lc].values)
    groups = np.array([str(p).replace("\\", "/").split("/")[0] for p in va[pc]])

    model = U.make_model(ck["backbone"], A, pretrained=False).to(dev)
    model.load_state_dict(ck["model"])
    dl = DataLoader(U.PARSet(U.resolve_paths(va[pc], a.img_root), y, m, U.build_tf(False, ck["h"], ck["w"])),
                    a.bs, shuffle=False, num_workers=a.workers)
    prob = U.predict_probs(model, dl, dev, tta=True)
    np.save(out / "val_probs.npy", prob)

    prior = np.tile((ytr * mtr).sum(0) / np.maximum(mtr.sum(0), 1), (len(y), 1))   # organizers' trivial baseline
    methods = {"Attribute prior (reference)": (prior, np.full(A, 0.5)),
               "Our model @0.5": (prob, np.full(A, 0.5)),
               "Our model + tuned thresholds": (prob, thr)}

    rows = []
    for g in list(np.unique(groups)) + ["MEAN over datasets"]:
        for mn, (p, t) in methods.items():
            r = U.gmetrics(p, y, m, t, groups) if g.startswith("MEAN") else \
                U.metrics(p[groups == g], y[groups == g], m[groups == g], t)
            rows.append(dict(dataset=g, method=mn, mA=round(r["mA"], 2), F1=round(r["F1"], 2), Avg=round(r["score"], 2)))
    dfd = pd.DataFrame(rows); dfd.to_csv(out / "per_dataset.csv", index=False)

    dfa = per_attribute(prob, y, m, thr, names).sort_values("mA")
    dfa.to_csv(out / "per_attribute.csv", index=False)

    # group attributes by family
    fam = lambda n: n.split("-")[0] if not n.startswith(("UpperBody", "LowerBody")) else "-".join(n.split("-")[:2])
    dff = dfa.assign(family=dfa.attribute.map(fam)).groupby("family")[["mA", "F1"]].mean().round(1).sort_values("mA")
    dff.to_csv(out / "per_family.csv")

    print("\n=== Per dataset ===\n" + dfd.to_string(index=False))
    print("\n=== 8 hardest attributes (lowest mA) ===\n", dfa.head(8).to_string(index=False))
    print("\n=== 5 easiest attributes ===\n", dfa.tail(5).to_string(index=False))
    print("\n=== By attribute family (mean mA / F1) ===\n", dff.to_string())
    json.dump(dict(thr=thr.tolist()), open(out / "thresholds.json", "w"))
    print(f"\nSaved CSVs to {out}")


if __name__ == "__main__":
    main()