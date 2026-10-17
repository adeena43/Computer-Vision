import shutil
from pathlib import Path
import numpy as np
from tqdm import tqdm
import download_datasets as d

data = Path("./data")

# ---- PA100k
pa = data / "PA100k"
pa.mkdir(parents=True, exist_ok=True)
if not (pa / "release_data" / "release_data").exists():
    print("Download PA100k dataset")
    d.gdown.download_folder("https://drive.google.com/drive/folders/1d_D0Yh7C262gr0ef9EqkvG_M3fqgAWa2?usp=sharing",
                            output=str(pa), quiet=False, use_cookies=False)
    print("Extract PA100k dataset")
    d.extract_zip(pa / "data.zip", pa)
else:
    print("PA100k already there, skipping")

# ---- PETA (Windows-safe paths)
peta = data / "PETA"
peta.mkdir(parents=True, exist_ok=True)
imgs = peta / "images"
if not imgs.exists() or not any(imgs.iterdir()):
    zf = peta / "peta.zip"
    print("Download PETA dataset")
    d.download_url("https://www.dropbox.com/s/52ylx522hwbdxz6/PETA.zip?dl=1", zf)
    print("Extract PETA dataset")
    d.extract_zip(zf, peta)
    imgs.mkdir(parents=True, exist_ok=True)
    mapping = {r[0]: r[1] for r in np.genfromtxt("peta_file_mapping.txt", dtype=str, delimiter=",")}
    moved = missing = 0
    for f in tqdm(list(peta.glob("*/*/*/*")), desc="Arranging PETA"):
        if f.suffix == ".txt" or not f.is_file():
            continue
        key = f.relative_to(data).as_posix()
        if key not in mapping:
            missing += 1
            continue
        shutil.move(str(f), str(data / mapping[key]))
        moved += 1
    print(f"PETA: moved {moved} files, {missing} not in mapping")
else:
    print("PETA already there, skipping")
print("DONE")