# tools/inspect_h5.py
import h5py, sys, re
from pathlib import Path

# 根路径
DATA_ROOT = Path("/home/lawrence/Desktop/algonauts-2025/algonauts2025/download/algonauts_2025.competitors")
SUBJECT   = "sub-01"

def walk(f):
    def _rec(g, pre=""):
        for k, v in g.items():
            p = f"{pre}/{k}" if pre else k
            if isinstance(v, h5py.Dataset):
                print(f"[D] {p:60s} shape={v.shape} attrs={dict(v.attrs)}")
            else:
                print(f"[G] {p:60s} attrs={dict(v.attrs)}")
                _rec(v, p)
    _rec(f)

def main():
    # 直接进入 func 文件夹
    fmri_dir = DATA_ROOT / "fmri" / SUBJECT / "func"
    files = sorted(fmri_dir.glob("*.h5"))
    if not files:
        print("No .h5 under", fmri_dir)
        sys.exit(1)

    print("Found H5 files:")
    for i, fp in enumerate(files, 1):
        print(f"  {i:02d}. {fp.name}")
    print("-"*80)

    fp0 = files[0]
    with h5py.File(fp0, "r") as f:
        print(f"== Inspect: {fp0.name} ==")
        print("File attrs:", dict(f.attrs))
        walk(f)

    # 从文件名中解析 task
    pat = re.compile(r"task-(?P<task>[^_]+)")
    for fp in files:
        m = pat.search(fp.name)
        print(f"PARSE {fp.name} ->", m.groupdict() if m else None)

if __name__ == "__main__":
    main()
