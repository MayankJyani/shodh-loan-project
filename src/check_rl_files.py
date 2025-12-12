import numpy as np
from pathlib import Path

out = Path("../data/rl")
for name in ["train_rl.npz", "val_rl.npz", "test_rl.npz"]:
    p = out / name
    print(f"\nChecking {p}")
    assert p.exists(), f"{p} missing"
    data = np.load(p)
    for k in data.files:
        arr = data[k]
        print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")