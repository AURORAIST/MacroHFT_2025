import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
# Target base path
base = Path(r'/home/ustc1958/lxy/10.20.2148MacroHFT底层六个/result/low_level/ETHUSDT/vol/1/label_3/seed_12345')
save = Path(r'/home/ustc1958/lxy/10.20.2148MacroHFT底层六个/tools')
out = {
    "base": str(base),
    "points": [],  # list of {epoch, val0, val1, mean}
}

if not base.exists():
    print(json.dumps({"error": f"base path not found: {base}"}, ensure_ascii=False))
    raise SystemExit(1)

# Collect epoch metrics
vals = []
for ep in sorted(base.glob('epoch_*'), key=lambda p: int(p.name.split('_')[1])):
    try:
        ep_num = int(ep.name.split('_')[1])
    except Exception:
        continue
    vdir = ep / 'val'
    v0_f = vdir / 'return_rate_mean_val_0.npy'
    v1_f = vdir / 'return_rate_mean_val_1.npy'
    if not (v0_f.exists() and v1_f.exists()):
        continue
    try:
        v0 = float(np.load(v0_f))
        v1 = float(np.load(v1_f))
        mean = (v0 + v1) / 2.0
        vals.append((ep_num, v0, v1, mean))
        out["points"].append({"epoch": ep_num, "val0": v0, "val1": v1, "mean": mean})
    except Exception as e:
        out.setdefault("skipped", []).append({"epoch": ep_num, "reason": str(e)})

if not vals:
    print(json.dumps({"error": "no validation metrics found"}, ensure_ascii=False))
    raise SystemExit(2)

# Compute diagnostics
vals_sorted = sorted(vals, key=lambda x: x[0])
epochs = [e for e, *_ in vals_sorted]
means = np.array([m for *_, m in vals_sorted], dtype=float)

best_idx = int(np.argmax(means))
best_epoch = epochs[best_idx]
best_val = float(means[best_idx])
last_val = float(means[-1])
last_epoch = int(epochs[-1])

# Convergence: last N window amplitude < tol
N = 5
conv_tol = 0.003
converged = False
if len(means) >= N:
    window = means[-N:]
    converged = (float(window.max() - window.min()) < conv_tol)

# Overfitting: last below best by margin and recent is non-increasing
margin = 0.01
recent = means[-N:] if len(means) >= N else means
non_increasing = bool(np.all(np.diff(recent) <= 1e-12)) if len(recent) > 1 else False
overfit = (last_val < (best_val - margin)) and non_increasing

summary = {
    "epochs": epochs,
    "val_mean": [float(x) for x in means.tolist()],
    "best_epoch": best_epoch,
    "best_val": best_val,
    "last_epoch": last_epoch,
    "last_val": last_val,
    "delta_last_to_best": float(last_val - best_val),
    "converged": converged,
    "overfit": overfit,
    "params": {"N": N, "conv_tol": conv_tol, "overfit_margin": margin}
}

print(json.dumps(summary, ensure_ascii=False))

# 可视化部分
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(epochs, [v[1] for v in vals_sorted], label='val0', marker='o')
plt.plot(epochs, [v[2] for v in vals_sorted], label='val1', marker='o')
plt.plot(epochs, means, label='mean', marker='o', linewidth=2)
plt.axvline(best_epoch, color='r', linestyle='--', label=f'Best Epoch {best_epoch}')
plt.scatter([best_epoch], [best_val], color='red', s=80, zorder=5)
plt.title('Validation Mean Curve')
plt.xlabel('Epoch')
plt.ylabel('Return Rate Mean')
plt.legend()

# 标注收敛/过拟合
text = f"Converged: {converged}\nOverfit: {overfit}\nBest: {best_val:.4f} @ {best_epoch}\nLast: {last_val:.4f} @ {last_epoch}"
plt.gca().text(0.02, 0.98, text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

# 保存图片
img_path = save / 'result.png'
plt.tight_layout()
plt.savefig(img_path)
print(f"Visualization saved to: {img_path}")
plt.close()