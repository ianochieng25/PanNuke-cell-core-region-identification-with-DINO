from pathlib import Path
import os, hashlib

src_hr = Path("data/tcia/raw/Custom/HR")
src_lr = Path("data/tcia/raw/Custom/LR_bicubic/X4")
dst_hr = Path("data/tcia/HR")
dst_lr = Path("data/tcia/LR_X4")
dst_hr.mkdir(parents=True, exist_ok=True)
dst_lr.mkdir(parents=True, exist_ok=True)
for p in src_hr.glob("*.png"):
    h = hashlib.md5(p.stem.encode()).hexdigest()[:6]
    os.link(p, dst_hr / f"{p.stem}_{h}.png")
for p in src_lr.glob("*.png"):
    h = hashlib.md5(p.stem.encode()).hexdigest()[:6]
    os.link(p, dst_lr / f"{p.stem}_{h}.png")
