"""
UPAR Challenge 2027 - Track 1 (Pedestrian Attribute Recognition)
Single-file pipeline: inspect -> train -> predict.

Usage (see README.md):
  python upar_par.py inspect --train_ann <file> --img_root <dir>
  python upar_par.py train   --train_ann <file> --val_ann <file> --img_root <dir> --out runs/exp1
  python upar_par.py predict --ckpt runs/exp1/best.pt --test_dir <dir> --template <csv> --out predictions.csv
"""
import argparse, json, os, random, time, copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import timm
from timm.utils import ModelEmaV2

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


# ----------------------------------------------------------------------------
# Annotation loading (format-agnostic: csv / tsv / pkl / json)
# ----------------------------------------------------------------------------
def load_table(path):
    p = str(path)
    if p.endswith(".pkl") or p.endswith(".pickle"):
        obj = pd.read_pickle(p)
        return obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)
    if p.endswith(".json"):
        return pd.read_json(p)
    sep = "\t" if p.endswith(".tsv") else None
    df = pd.read_csv(p, sep=sep, engine="python")
    # headerless file? (first row numeric-looking in all but first col)
    return df


def split_table(df, path_col=None, label_cols=None):
    """Return (path_col, label_cols). Heuristic: path col = first object column
    whose values look like file names; labels = all remaining numeric cols."""
    if path_col is None:
        cands = [c for c in df.columns if df[c].dtype == object]
        for c in cands:
            s = str(df[c].iloc[0]).lower()
            if any(s.endswith(e) for e in IMG_EXT) or "/" in s or "\\" in s:
                path_col = c
                break
        if path_col is None:
            path_col = cands[0] if cands else df.columns[0]
    if label_cols is None:
        label_cols = [c for c in df.columns
                      if c != path_col and pd.api.types.is_numeric_dtype(df[c])]
        # drop id-like columns that are not binary-ish
        keep = []
        for c in label_cols:
            u = set(pd.unique(df[c].dropna()))
            if len(u) <= 4 and u.issubset({0, 1, 2, -1, 0.0, 1.0, 2.0, -1.0}):
                keep.append(c)
        label_cols = keep
    return path_col, label_cols


def clean_labels(arr):
    """Return (y in {0,1}, mask 1=known). Anything not 0/1 (e.g. 2 or -1 = unknown) is masked."""
    arr = np.asarray(arr, dtype=np.float32)
    mask = ((arr == 0) | (arr == 1)).astype(np.float32)
    y = np.where(mask > 0, arr, 0).astype(np.float32)
    return y, mask


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
class LowRes:  # simulate low-quality surveillance crops (module level => picklable on Windows)
    def __call__(self, im):
        if random.random() < 0.3:
            s = random.uniform(0.3, 0.7)
            im = im.resize((max(8, int(im.width * s)), max(16, int(im.height * s))))
        return im


def build_tf(train, h, w):
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if not train:
        return T.Compose([T.Resize((h, w)), T.ToTensor(), norm])

    return T.Compose([
        LowRes(),
        T.Resize((h, w)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.3, 0.05)], p=0.8),
        T.RandomApply([T.GaussianBlur(5, (0.1, 1.5))], p=0.2),
        T.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        T.ToTensor(), norm,
        T.RandomErasing(p=0.4, scale=(0.02, 0.2)),
    ])


class PARSet(Dataset):
    def __init__(self, paths, y, mask, tf):
        self.paths, self.y, self.mask, self.tf = paths, y, mask, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB")
        return self.tf(im), self.y[i], self.mask[i]


def resolve_paths(names, img_root):
    out = []
    root = Path(img_root)
    for n in names:
        n = str(n).replace("\\", "/")
        p = Path(n)
        out.append(str(p if p.is_absolute() and p.exists() else root / n))
    return out


# ----------------------------------------------------------------------------
# Metrics: label-based mA, instance-based F1, challenge score = harmonic mean
# ----------------------------------------------------------------------------
def metrics(prob, y, mask, thr):
    pred = (prob >= thr[None, :]).astype(np.float32)
    m = mask > 0
    # mA
    tp = (pred * y * m).sum(0); pos = (y * m).sum(0)
    tn = ((1 - pred) * (1 - y) * m).sum(0); neg = ((1 - y) * m).sum(0)
    tpr = np.where(pos > 0, tp / np.maximum(pos, 1), np.nan)
    tnr = np.where(neg > 0, tn / np.maximum(neg, 1), np.nan)
    ma = np.nanmean(np.nanmean(np.stack([tpr, tnr]), 0))
    # instance F1 (only over known attributes)
    P = pred * m; G = y * m
    inter = (P * G).sum(1); ps = P.sum(1); gs = G.sum(1)
    prec = np.where(ps > 0, inter / np.maximum(ps, 1), 0)
    rec = np.where(gs > 0, inter / np.maximum(gs, 1), 0)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-9), 0).mean()
    score = 2 * ma * f1 / max(ma + f1, 1e-9)
    return dict(mA=float(ma * 100), F1=float(f1 * 100), score=float(score * 100))


def gmetrics(prob, y, mask, thr, groups):
    """Challenge protocol: score every dataset on its own, then average over datasets."""
    rs = [metrics(prob[groups == g], y[groups == g], mask[groups == g], thr) for g in np.unique(groups)]
    return {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}


def tune_thresholds(prob, y, mask, groups=None, iters=2, grid=np.linspace(0.1, 0.9, 17), base=0.5):
    """Coordinate ascent on per-attribute thresholds to maximise the challenge score."""
    A = prob.shape[1]
    thr = np.full(A, base, dtype=np.float32)
    if groups is None:
        groups = np.zeros(len(prob), dtype=int)
    best = gmetrics(prob, y, mask, thr, groups)["score"]
    for _ in range(iters):
        for a in range(A):
            keep = thr[a]
            for g in grid:
                thr[a] = g
                s = gmetrics(prob, y, mask, thr, groups)["score"]
                if s > best + 1e-6:
                    best, keep = s, g
            thr[a] = keep
    return thr


# ----------------------------------------------------------------------------
# Model / loss
# ----------------------------------------------------------------------------
def make_model(backbone, n_attr, pretrained=True, drop=0.2):
    if backbone.startswith("tv_"):  # torchvision model: runs in the evaluation container (no timm there)
        import torchvision
        kw = dict(stochastic_depth_prob=0.1) if "convnext" in backbone else {}
        m = getattr(torchvision.models, backbone[3:])(weights="DEFAULT" if pretrained else None, **kw)
        last = m.classifier[-1]
        m.classifier[-1] = nn.Sequential(nn.Dropout(drop), nn.Linear(last.in_features, n_attr))
        return m
    return timm.create_model(backbone, pretrained=pretrained, num_classes=n_attr,
                             drop_rate=drop, drop_path_rate=0.1)


def masked_bce(logits, y, mask, pos_w, smooth=0.02):
    y = y * (1 - smooth) + 0.5 * smooth
    loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    w = torch.where(y > 0.5, pos_w[None, :], torch.ones_like(pos_w)[None, :])
    return (loss * w * mask).sum() / mask.sum().clamp(min=1)


@torch.no_grad()
def predict_probs(model, loader, device, tta=True):
    model.eval()
    ps = []
    for x, *_ in tqdm(loader, leave=False):
        x = x.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            p = torch.sigmoid(model(x).float())
            if tta:
                p = (p + torch.sigmoid(model(torch.flip(x, dims=[3])).float())) / 2
        ps.append(p.cpu())
    return torch.cat(ps).numpy()


# ----------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------
def cmd_inspect(a):
    df = load_table(a.train_ann)
    print("shape:", df.shape)
    print("columns:", list(df.columns)[:60])
    print(df.head(3).to_string()[:1500])
    pc, lc = split_table(df, a.path_col)
    print(f"\n-> detected path column: {pc!r}; {len(lc)} label columns")
    if lc:
        vals = pd.unique(df[lc].values.ravel())
        print("label values present:", sorted(vals.tolist())[:10])
    names = df[pc].head(5).tolist()
    ps = resolve_paths(names, a.img_root)
    print("sample image paths exist?:", [os.path.exists(p) for p in ps])
    print("example:", ps[0])


def cmd_train(a):
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    tr = load_table(a.train_ann); va = load_table(a.val_ann)
    pc, lc = split_table(tr, a.path_col, a.label_cols.split(",") if a.label_cols else None)
    ytr, mtr = clean_labels(tr[lc].values); yva, mva = clean_labels(va[lc].values)
    A = len(lc)
    print(f"train {len(tr)}  val {len(va)}  attributes {A}  device {device}")
    if a.limit:
        tr = tr.iloc[:a.limit]; ytr, mtr = ytr[:a.limit], mtr[:a.limit]
        va = va.iloc[:a.limit]; yva, mva = yva[:a.limit], mva[:a.limit]

    # class-imbalance weights from train frequency (sqrt-dampened)
    pos = (ytr * mtr).sum(0); known = mtr.sum(0)
    freq = np.clip(pos / np.maximum(known, 1), 1e-3, 1 - 1e-3)
    pos_w = torch.tensor(np.clip(np.sqrt((1 - freq) / freq), 1.0, 8.0), dtype=torch.float32, device=device)

    dtr = PARSet(resolve_paths(tr[pc], a.img_root), ytr, mtr, build_tf(True, a.h, a.w))
    gva = np.array([str(p).replace("\\", "/").split("/")[0] for p in va[pc]])
    print("val datasets:", dict(zip(*np.unique(gva, return_counts=True))))
    dva = PARSet(resolve_paths(va[pc], a.img_root), yva, mva, build_tf(False, a.h, a.w))
    ltr = DataLoader(dtr, a.bs, shuffle=True, num_workers=a.workers, pin_memory=True, drop_last=True,
                     persistent_workers=a.workers > 0)
    lva = DataLoader(dva, a.bs * 2, shuffle=False, num_workers=a.workers, pin_memory=True)

    model = make_model(a.backbone, A, pretrained=not a.no_pretrained).to(device)
    ema = ModelEmaV2(model, decay=a.ema)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    total = a.epochs * len(ltr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.1)

    meta = dict(attributes=lc, backbone=a.backbone, h=a.h, w=a.w, path_col=pc)
    best = -1; log = []
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); run = 0
        for i, (x, y, m) in enumerate(tqdm(ltr, desc=f"ep{ep+1}/{a.epochs}", leave=False)):
            x, y, m = x.to(device, non_blocking=True), y.to(device), m.to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(x)
            loss = masked_bce(logits.float(), y, m, pos_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step(); ema.update(model)
            run += loss.item()
        prob = predict_probs(ema.module, lva, device, tta=False)
        r = gmetrics(prob, yva, mva, np.full(A, 0.5), gva)
        r.update(epoch=ep + 1, loss=run / len(ltr), sec=round(time.time() - t0))
        log.append(r); print(r)
        if r["score"] > best:
            best = r["score"]
            torch.save(dict(model=ema.module.state_dict(), **meta), out / "best.pt")
        json.dump(log, open(out / "log.json", "w"), indent=1)

    # final: per-attribute thresholds tuned on val with the best EMA checkpoint
    ck = torch.load(out / "best.pt", map_location=device)
    ema.module.load_state_dict(ck["model"])
    prob = predict_probs(ema.module, lva, device, tta=True)
    base = gmetrics(prob, yva, mva, np.full(A, 0.5), gva)
    thr = tune_thresholds(prob, yva, mva, gva)
    tuned = gmetrics(prob, yva, mva, thr, gva)
    print("val @0.5+TTA:", base); print("val tuned thr:", tuned)
    ck["thr"] = thr.tolist()
    torch.save(ck, out / "best.pt")
    json.dump(dict(base=base, tuned=tuned, thr=thr.tolist()), open(out / "final.json", "w"), indent=1)


def cmd_predict(a):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location=device)
    lc = ck["attributes"]; A = len(lc)
    model = make_model(ck["backbone"], A, pretrained=False).to(device)
    model.load_state_dict(ck["model"])
    thr = np.array(ck.get("thr", [0.5] * A), dtype=np.float32)
    if a.no_thr:
        thr[:] = 0.5

    tmpl = load_table(a.template) if a.template else None
    if tmpl is not None:
        pc = a.path_col or tmpl.columns[0]
        names = tmpl[pc].astype(str).tolist()
    else:
        names = sorted(str(p.relative_to(a.test_dir)) for p in Path(a.test_dir).rglob("*")
                       if p.suffix.lower() in IMG_EXT)
        pc = "image"
    paths = resolve_paths(names, a.test_dir)
    dummy = np.zeros((len(paths), A), np.float32)
    dl = DataLoader(PARSet(paths, dummy, dummy, build_tf(False, ck["h"], ck["w"])),
                    a.bs, shuffle=False, num_workers=a.workers)
    prob = predict_probs(model, dl, device, tta=True)
    pred = (prob >= thr[None]).astype(int)

    out = pd.DataFrame(pred, columns=lc)
    if tmpl is not None:
        res = tmpl.copy()
        tcols = [c for c in tmpl.columns if c != pc]
        if set(lc) <= set(tcols):  # same column names
            for c in lc:
                res[c] = out[c].values
        elif len(tcols) == A:      # same count, fill by position
            for c, l in zip(tcols, lc):
                res[c] = out[l].values
        else:
            print(f"WARNING: template has {len(tcols)} label cols, model has {A}. Writing generic file.")
            res = pd.concat([tmpl[[pc]], out], axis=1)
        res.to_csv(a.out, index=False, header=a.header)
    else:
        pd.concat([pd.Series(names, name=pc), out], axis=1).to_csv(a.out, index=False, header=a.header)
    print("wrote", a.out, "rows", len(paths))


RUN_PY = r"""# UPAR 2027 Track 1 submission. Only torch/torchvision/numpy/Pillow are used (no timm).
# Per-attribute thresholds are folded in so that "probability > 0.5" == "attribute present".
from __future__ import annotations
import io, json, os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
_M = {}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
POOL = ThreadPoolExecutor(8)
BATCH_SIZE = 128


def _build(backbone, A):
    m = getattr(torchvision.models, backbone[3:])(weights=None)
    last = m.classifier[-1]
    m.classifier[-1] = nn.Sequential(nn.Dropout(0.0), nn.Linear(last.in_features, A))
    return m


def load_model():
    meta = json.load(open(os.path.join(HERE, "assets", "meta.json")))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = _build(meta["backbone"], len(meta["attributes"]))
    sd = torch.load(os.path.join(HERE, "assets", "model.pt"), map_location="cpu", weights_only=True)
    m.load_state_dict({k: v.float() for k, v in sd.items()})
    m = m.to(dev).eval()
    if dev == "cuda":
        m = m.to(memory_format=torch.channels_last)
    thr = np.clip(np.array(meta["thr"], np.float32), 1e-3, 1 - 1e-3)
    _M.update(model=m, dev=dev, meta=meta, shift=torch.tensor(np.log(thr / (1 - thr)), device=dev))


def _to_pil(x):
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    if isinstance(x, (bytes, bytearray)):
        return Image.open(io.BytesIO(x)).convert("RGB")
    if isinstance(x, (str, os.PathLike)):
        return Image.open(x).convert("RGB")
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[0] in (1, 3) and x.shape[-1] not in (1, 3):
        x = x.transpose(1, 2, 0)
    if x.dtype != np.uint8:
        x = (x * 255).clip(0, 255).astype(np.uint8) if x.max() <= 1.0 else x.astype(np.uint8)
    return Image.fromarray(x).convert("RGB")


def _get_image(sample):
    if not isinstance(sample, dict):
        return _to_pil(sample)
    for k in ("image_path", "image", "img", "pil_image", "image_array", "array", "path", "file", "filename"):
        if k in sample and sample[k] is not None:
            return _to_pil(sample[k])
    raise KeyError(f"no image field in sample; keys = {list(sample.keys())}")


def _prep(sample, h, w):
    im = _get_image(sample).resize((w, h), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def predict_batch(samples):
    if not _M:
        load_model()
    meta, dev = _M["meta"], _M["dev"]
    h, w = meta["h"], meta["w"]
    arr = np.stack(list(POOL.map(lambda s: _prep(s, h, w), samples)))
    x = torch.from_numpy(arr).to(dev).permute(0, 3, 1, 2).float().div_(255)
    x = ((x - MEAN.to(dev)) / STD.to(dev)).contiguous(memory_format=torch.channels_last)
    out = []
    with torch.inference_mode(), torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
        for i in range(0, len(x), 128):
            xb = x[i:i + 128]
            p = (torch.sigmoid(_M["model"](xb).float()) + torch.sigmoid(_M["model"](torch.flip(xb, dims=[3])).float())) / 2
            p = p.clamp(1e-6, 1 - 1e-6)
            out.append(torch.sigmoid(torch.log(p / (1 - p)) - _M["shift"]).cpu())
    probs = torch.cat(out).numpy()
    names = meta["attributes"]
    if samples and isinstance(samples[0], dict) and samples[0].get("attribute_names"):
        idx = [names.index(n) if n in names else -1 for n in samples[0]["attribute_names"]]
        probs = np.stack([probs[:, j] if j >= 0 else np.zeros(len(probs), np.float32) for j in idx], 1)
    probs = np.nan_to_num(probs, nan=0.0).clip(0.0, 1.0)
    return [row.tolist() for row in probs]
"""

META_YAML = """# Code submissions are executed by the challenges ingestion program, which
# imports run.py and calls predict_image() / predict_batch(). Nothing in this
# file is used for inference, it exists only so the platform never treats the
# archive as a result submission. Leave it as-is.
command: python3 -c "print('Submission package ready. The ingestion program imports run.py')"
"""


def cmd_package(a):
    import shutil, zipfile
    ck = torch.load(a.ckpt, map_location="cpu")
    assert ck["backbone"].startswith("tv_"), "package needs a torchvision backbone (tv_*): timm is not in the container"
    A = len(ck["attributes"])
    sub = Path(a.out_dir)
    if sub.exists():
        shutil.rmtree(sub)
    (sub / "assets").mkdir(parents=True)
    sd = {k: (v.half() if v.is_floating_point() else v) for k, v in ck["model"].items()}
    torch.save(sd, sub / "assets" / "model.pt")
    json.dump(dict(attributes=ck["attributes"], thr=ck.get("thr", [0.5] * A), h=ck["h"], w=ck["w"],
                   backbone=ck["backbone"]), open(sub / "assets" / "meta.json", "w"))
    (sub / "run.py").write_text(RUN_PY); (sub / "metadata.yaml").write_text(META_YAML)
    zp = Path(a.out_zip)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:  # files at zip ROOT (no parent folder)
        for f in sub.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(sub).as_posix())
    print(f"wrote {zp}  ({zp.stat().st_size/1e6:.1f} MB)")


def cmd_selftest(a):
    """Mimic the ingestion program: import run.py from the package, predict val images, score."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("run", str(Path(a.sub_dir) / "run.py"))
    run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)
    va = load_table(a.val_ann)
    pc, lc = split_table(va, a.path_col)
    va = va.sample(n=min(a.n, len(va)), random_state=0).reset_index(drop=True)
    y, m = clean_labels(va[lc].values)
    paths = resolve_paths(va[pc], a.img_root)
    t0 = time.time(); probs = []
    for i in range(0, len(paths), 64):
        samples = [dict(image=Image.open(p).convert("RGB"), image_path=p, attribute_names=lc) for p in paths[i:i+64]]
        probs += run.predict_batch(samples)
    dt = time.time() - t0
    probs = np.array(probs)
    print("shape", probs.shape, f"| {len(paths)/dt:.1f} img/s on {'cuda' if torch.cuda.is_available() else 'cpu'}")
    g = np.array([str(p).replace("\\", "/").split("/")[0] for p in va[pc]])
    print("selftest @0.5 (thresholds already folded in):", gmetrics(probs, y, m, np.full(len(lc), 0.5), g))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--path_col", default=None)
        p.add_argument("--img_root", default=".")

    p = sp.add_parser("inspect"); common(p); p.add_argument("--train_ann", required=True)
    p = sp.add_parser("train"); common(p)
    p.add_argument("--train_ann", required=True); p.add_argument("--val_ann", required=True)
    p.add_argument("--label_cols", default=None, help="comma separated; default auto-detect")
    p.add_argument("--backbone", default="tv_convnext_small")
    p.add_argument("--h", type=int, default=256); p.add_argument("--w", type=int, default=128)
    p.add_argument("--epochs", type=int, default=12); p.add_argument("--bs", type=int, default=48)
    p.add_argument("--lr", type=float, default=3e-4 / 3); p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--workers", type=int, default=8); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_pretrained", action="store_true"); p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default="runs/exp1")
    p = sp.add_parser("predict"); common(p)
    p.add_argument("--ckpt", required=True); p.add_argument("--test_dir", required=True)
    p.add_argument("--template", default=None); p.add_argument("--out", default="predictions.csv")
    p.add_argument("--bs", type=int, default=128); p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no_thr", action="store_true")
    p.add_argument("--header", type=lambda s: s.lower() != "false", default=True)
    p = sp.add_parser("package")
    p.add_argument("--ckpt", required=True); p.add_argument("--out_dir", default="submission")
    p.add_argument("--out_zip", default="submission.zip"); p.add_argument("--freeze", action="store_true")
    p = sp.add_parser("selftest"); common(p)
    p.add_argument("--sub_dir", default="submission"); p.add_argument("--val_ann", required=True)
    p.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    {"inspect": cmd_inspect, "train": cmd_train, "predict": cmd_predict,
     "package": cmd_package, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
