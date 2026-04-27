import os
import numpy as np
import sklearn.metrics as skm
import json

ROOT = os.path.join(os.path.dirname(__file__), '..')
OUT_PRED = os.path.join(ROOT, 'output', 'isruc_predict')

fold_metrics = []
for f in range(10):
    pdir = os.path.join(OUT_PRED, f'fold_{f}')
    npz_path = os.path.join(pdir, 'isruc.npz')
    if not os.path.exists(npz_path):
        print(f'skip fold {f}, missing {npz_path}')
        continue
    data = np.load(npz_path, allow_pickle=True)
    y_true = np.array(data['y_true'])
    y_pred = np.array(data['y_pred'])
    acc = skm.accuracy_score(y_true, y_pred)
    mf1 = skm.f1_score(y_true, y_pred, average='macro')
    fold_metrics.append((acc, mf1))
    print(f'fold {f}: acc={acc:.4f}, mf1={mf1:.4f}')

if len(fold_metrics) == 0:
    print('No folds found. Check predictions under', OUT_PRED)
    raise SystemExit(1)

accs = np.array([m[0] for m in fold_metrics])
mf1s = np.array([m[1] for m in fold_metrics])

summary = {
    'n_folds': int(len(fold_metrics)),
    'acc_mean': float(accs.mean()),
    'acc_std': float(accs.std()),
    'mf1_mean': float(mf1s.mean()),
    'mf1_std': float(mf1s.std()),
}

out_path = os.path.join(os.path.dirname(OUT_PRED), 'isruc_cv_summary.json')
with open(out_path, 'w') as f:
    json.dump(summary, f, indent=2)

print('Summary saved to', out_path)
print(json.dumps(summary, indent=2))
