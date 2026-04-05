"""
Quick comparison script: trains (or loads) the SmallFundusCNN and AdaBoost pipelines on the same dataset
and prints side-by-side metrics (F1, AUC, confusion matrix) for validation and/or test splits.

Intended usage (PowerShell examples):

# Train both from scratch (light epochs for CNN) and compare on validation split
python compare_models.py --data_root data_root --imgsz 224 --epochs 3 --batch 16 --run_mode train

# Evaluate using existing saved CNN (runs/cnn_best.pt) and AdaBoost (runs/adaboost.joblib)
python compare_models.py --data_root data_root --imgsz 224 --run_mode eval

Notes:
- CNN training can take time; keep epochs small for quick comparisons.
- AdaBoost trains quickly on CPU.
- For reproducibility you can set --seed.
- This script assumes the directory structure data_root/{train,val,test}/{class_name}/images.
"""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, precision_recall_fscore_support, f1_score

# Reuse existing modules
from models.cnn_model import SmallFundusCNN, CLAHEDataset
from models.adaboost_model import load_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import dump, load

from torch.utils.data import DataLoader
from torchvision import transforms


def build_cnn_loaders(root: Path, imgsz: int, batch: int, workers: int = 0):
    norm = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        norm
    ])
    base = transforms.Compose([norm])
    ds_train = CLAHEDataset(root/'train', transform=aug, size=imgsz)
    ds_val   = CLAHEDataset(root/'val', transform=base, size=imgsz)
    ds_test  = CLAHEDataset(root/'test', transform=base, size=imgsz)
    pin_mem = torch.cuda.is_available()
    loaders = {
        'train': DataLoader(ds_train, batch_size=batch, shuffle=True,  num_workers=workers, pin_memory=pin_mem),
        'val':   DataLoader(ds_val,   batch_size=batch, shuffle=False, num_workers=workers, pin_memory=pin_mem),
        'test':  DataLoader(ds_test,  batch_size=batch, shuffle=False, num_workers=workers, pin_memory=pin_mem),
    }
    return loaders, len(ds_train.classes)


def train_cnn(model, loader, device, optimizer, criterion):
    model.train(); loss_sum=0; correct=0; total=0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits,y)
        loss.backward(); optimizer.step()
        loss_sum += loss.item()*x.size(0)
        pred = logits.argmax(1)
        correct += (pred==y).sum().item(); total += x.size(0)
    return loss_sum/total, correct/total


def eval_cnn(model, loader, device):
    model.eval(); all_logits=[]; all_y=[]
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device); logits=model(x)
            all_logits.append(logits.cpu()); all_y.append(y)
    logits=torch.cat(all_logits); y=torch.cat(all_y)
    probs = F.softmax(logits, dim=1).numpy(); y_np=y.numpy(); preds=probs.argmax(1)
    p,r,f,_ = precision_recall_fscore_support(y_np, preds, average='macro', zero_division=0)
    if probs.shape[1] == 2:
        try:
            auc = roc_auc_score(y_np, probs[:,1])
        except Exception:
            auc = float('nan')
    else:
        try:
            auc = roc_auc_score(y_np, probs, multi_class='ovr')
        except Exception:
            auc = float('nan')
    report = classification_report(y_np, preds, zero_division=0)
    cm = confusion_matrix(y_np, preds)
    return f, auc, report, cm


def train_adaboost(root: Path, imgsz: int, out_path: Path, n_estimators: int = 200, learning_rate: float = 0.5):
    X, y, _ = load_split(root/'train', out_size=imgsz)
    base = DecisionTreeClassifier(max_depth=2)
    # Handle scikit-learn API change: 'estimator' vs 'base_estimator'
    try:
        clf = AdaBoostClassifier(estimator=base, n_estimators=n_estimators, learning_rate=learning_rate, algorithm='SAMME.R', random_state=42)
    except TypeError:
        clf = AdaBoostClassifier(base_estimator=base, n_estimators=n_estimators, learning_rate=learning_rate, algorithm='SAMME.R', random_state=42)
    pipe = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),
        ('clf', clf)
    ])
    pipe.fit(X,y)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dump(pipe, out_path)
    return pipe


def eval_adaboost(pipe, root: Path, split: str, imgsz: int):
    X, y, _ = load_split(root/split, out_size=imgsz)
    probs = pipe.predict_proba(X)
    preds = probs.argmax(axis=1)
    f1 = f1_score(y, preds, average='macro')
    try:
        if probs.shape[1] == 2:
            auc = roc_auc_score(y, probs[:,1])
        else:
            auc = roc_auc_score(y, probs, multi_class='ovr')
    except Exception:
        auc = float('nan')
    report = classification_report(y, preds, zero_division=0)
    cm = confusion_matrix(y, preds)
    return f1, auc, report, cm


def main(args):
    root = Path(args.data_root)
    for split in ['train','val','test']:
        if not (root/split).exists():
            raise SystemExit(f"Missing split folder: {root/split}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaders, num_classes = build_cnn_loaders(root, args.imgsz, args.batch, workers=args.workers)
    cnn_ckpt_path = Path('runs')/'cnn_best.pt'
    ada_path = Path('runs')/'adaboost.joblib'

    # CNN pipeline
    if args.run_mode == 'train':
        model = SmallFundusCNN(num_classes=num_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        criterion = torch.nn.CrossEntropyLoss()
        best_f1 = 0.0
        for epoch in range(1, args.epochs+1):
            t0=time.time(); tr_loss, tr_acc = train_cnn(model, loaders['train'], device, optimizer, criterion)
            f1, auc, report, cm = eval_cnn(model, loaders['val'], device)
            dt=time.time()-t0
            print(f"[CNN] Epoch {epoch}/{args.epochs} loss {tr_loss:.4f} acc {tr_acc:.3f} f1 {f1:.3f} auc {auc:.3f} time {dt:.1f}s")
            if f1 > best_f1:
                best_f1 = f1
                torch.save({'model': model.state_dict(), 'num_classes': num_classes}, cnn_ckpt_path)
                print(f"[CNN] Saved best ckpt to {cnn_ckpt_path}")
        print("[CNN] Training complete.")
    else:
        if not cnn_ckpt_path.exists():
            raise SystemExit("CNN checkpoint not found; run with --run_mode train first.")
        ckpt = torch.load(cnn_ckpt_path, map_location=device)
        model = SmallFundusCNN(num_classes=ckpt['num_classes']).to(device)
        model.load_state_dict(ckpt['model'])

    # AdaBoost pipeline
    if args.run_mode == 'train':
        pipe = train_adaboost(root, args.imgsz, ada_path)
        print(f"[AdaBoost] Saved model to {ada_path}")
    else:
        if not ada_path.exists():
            raise SystemExit("AdaBoost model not found; run with --run_mode train first.")
        pipe = load(ada_path)

    # Evaluate both on requested split (default val unless eval mode then test)
    eval_split = args.eval_split
    print(f"\n=== Evaluation Split: {eval_split} ===")
    cnn_f1, cnn_auc, cnn_report, cnn_cm = eval_cnn(model, loaders[eval_split], device)
    ada_f1, ada_auc, ada_report, ada_cm = eval_adaboost(pipe, root, eval_split, args.imgsz)

    print("\n[CNN] Classification report:\n" + cnn_report)
    print("[CNN] Confusion matrix:\n" + str(cnn_cm))
    print(f"[CNN] F1 {cnn_f1:.3f} | AUC {cnn_auc:.3f}")

    print("\n[AdaBoost] Classification report:\n" + ada_report)
    print("[AdaBoost] Confusion matrix:\n" + str(ada_cm))
    print(f"[AdaBoost] F1 {ada_f1:.3f} | AUC {ada_auc:.3f}")

    # Simple side-by-side summary
    print("\n=== Summary ===")
    print(f"Model,Split,{eval_split},F1,AUC")
    print(f"CNN,{eval_split},{cnn_f1:.4f},{cnn_auc:.4f}")
    print(f"AdaBoost,{eval_split},{ada_f1:.4f},{ada_auc:.4f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, required=True)
    ap.add_argument('--imgsz', type=int, default=224)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=3, help='CNN epochs (AdaBoost trains once)')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--run_mode', choices=['train','eval'], default='train', help='Train both or just evaluate existing checkpoints')
    ap.add_argument('--eval_split', choices=['val','test'], default='val', help='Which split to evaluate for side-by-side metrics')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    main(args)
