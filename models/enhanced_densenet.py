"""
Enhanced DenseNet for DR lesion-focused classification
Additions over standard DenseNet:
- Squeeze-and-Excitation (SE) blocks for channel attention
- Multi-scale feature fusion (concatenate intermediate block outputs)
- Lesion-focused preprocessing (CLAHE + center crop)

Usage:
python enhanced_densenet.py --data_root path/to/dataset --epochs 15 --imgsz 256
"""
import os
import argparse
from pathlib import Path
import time
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import torchvision.models as tvm
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import cv2
from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

# --------- Preprocessing ---------

def apply_clahe_rgb(img_bgr, clip=2.0, tile=(8,8)):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    cl = clahe.apply(l)
    merged = cv2.merge((cl,a,b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def center_crop_square(img):
    h,w = img.shape[:2]
    m=min(h,w)
    y0=(h-m)//2; x0=(w-m)//2
    return img[y0:y0+m, x0:x0+m]


def preprocess_image(img_bgr, size=256):
    img = apply_clahe_rgb(img_bgr)
    img = center_crop_square(img)
    img = cv2.resize(img, (size,size), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

class CLAHEDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None, size=256):
        super().__init__(root)
        self.tf = transform
        self.size = size
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            raise FileNotFoundError(path)
        img = preprocess_image(img_bgr, self.size)
        tensor = transforms.ToTensor()(img)
        if self.tf:
            tensor = self.tf(tensor)
        return tensor, target

# --------- SE Block ---------
class SEBlock(nn.Module):
    def __init__(self, c: int, r: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c, c//r),
            nn.ReLU(inplace=True),
            nn.Linear(c//r, c),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(x)
        w = w.view(x.size(0), x.size(1), 1,1)
        return x * w

# --------- Enhanced DenseNet Wrapper ---------
class EnhancedDenseNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        base = tvm.densenet121(weights=None)  # no pretrained to keep portability
        self.features = base.features
        # Insert SE blocks after denseblock outputs
        self.se1 = SEBlock(256)
        self.se2 = SEBlock(512)
        self.se3 = SEBlock(1024)
        self.se4 = SEBlock(1024)
        self.classifier = nn.Sequential(
            nn.Linear(256+512+1024+1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Mirror densenet feature flow
        feats: List[torch.Tensor] = []
        for name, module in self.features._modules.items():
            x = module(x)
            if name == 'denseblock1':
                x = self.se1(x); feats.append(F.adaptive_avg_pool2d(x,1).flatten(1))
            elif name == 'denseblock2':
                x = self.se2(x); feats.append(F.adaptive_avg_pool2d(x,1).flatten(1))
            elif name == 'denseblock3':
                x = self.se3(x); feats.append(F.adaptive_avg_pool2d(x,1).flatten(1))
            elif name == 'denseblock4':
                x = self.se4(x); feats.append(F.adaptive_avg_pool2d(x,1).flatten(1))
        fused = torch.cat(feats, dim=1)
        return self.classifier(fused)

# --------- Train / Eval ---------

def get_loaders(root: Path, size: int, batch: int, workers: int=0, use_balanced_sampler: bool=False):
    norm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        norm
    ])
    base = transforms.Compose([norm])
    ds_train = CLAHEDataset(root/'train', aug, size)
    ds_val   = CLAHEDataset(root/'val', base, size)
    ds_test  = CLAHEDataset(root/'test', base, size)

    # Only pin memory when CUDA is available to avoid PyTorch warnings on CPU-only machines
    pin_mem = torch.cuda.is_available()

    if use_balanced_sampler:
        train_targets = ds_train.targets
        class_counts = np.bincount(train_targets)
        class_counts[class_counts == 0] = 1
        sample_weights = np.array([1.0 / class_counts[t] for t in train_targets], dtype=np.float32)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(ds_train, batch_size=batch, sampler=sampler, num_workers=workers, pin_memory=pin_mem)
    else:
        train_loader = DataLoader(ds_train, batch_size=batch, shuffle=True,  num_workers=workers, pin_memory=pin_mem)

    loaders = {
        'train': train_loader,
        'val':   DataLoader(ds_val,   batch_size=batch, shuffle=False, num_workers=workers, pin_memory=pin_mem),
        'test':  DataLoader(ds_test,  batch_size=batch, shuffle=False, num_workers=workers, pin_memory=pin_mem),
    }
    return loaders, len(ds_train.classes)


def train_epoch(model, loader, device, opt, criterion):
    model.train(); loss_sum=0; correct=0; total=0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward(); opt.step()
        loss_sum += loss.item()*x.size(0)
        pred = logits.argmax(1)
        correct += (pred==y).sum().item(); total += x.size(0)
    return loss_sum/total, correct/total


def evaluate(model, loader, device):
    model.eval(); all_logits=[]; all_y=[]
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device); logits=model(x)
            all_logits.append(logits.cpu()); all_y.append(y)
    logits=torch.cat(all_logits); y=torch.cat(all_y)
    probs = F.softmax(logits, dim=1).numpy(); y_np=y.numpy(); preds=probs.argmax(1)
    # Use zero_division=0 to avoid UndefinedMetricWarning when some classes have no predicted samples
    p, r, f, sup = precision_recall_fscore_support(y_np, preds, average='macro', zero_division=0)
    f1 = f
    try:
        if probs.shape[1] > 2:
            auc = roc_auc_score(y_np, probs, multi_class='ovr')
        else:
            auc = roc_auc_score(y_np, probs[:,1])
    except Exception:
        auc = float('nan')
    report = classification_report(y_np, preds, zero_division=0)
    cm = confusion_matrix(y_np, preds)
    pred_counts = np.bincount(preds, minlength=probs.shape[1])
    if (pred_counts == 0).any():
        print("Warning: at least one class has zero predicted samples. Pred counts:", pred_counts)
        print("True label distribution:", np.bincount(y_np, minlength=probs.shape[1]))
    return f1, auc, report, cm


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaders, n_classes = get_loaders(Path(args.data_root), args.imgsz, args.batch, args.workers, use_balanced_sampler=args.balanced_sampler)
    model = EnhancedDenseNet(num_classes=n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Optionally compute class weights from the training set and pass to CrossEntropyLoss
    if args.use_class_weights:
        train_targets = loaders['train'].dataset.targets
        class_counts = np.bincount(train_targets)
        class_counts[class_counts == 0] = 1
        total = float(len(train_targets))
        weights = total / (len(class_counts) * class_counts.astype(np.float32))
        weights = torch.tensor(weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print('Using class weights for loss:', weights.cpu().numpy())
    else:
        criterion = nn.CrossEntropyLoss()
    best_f1=0.0; out_dir=Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    best_path=out_dir/'enhanced_densenet.pt'

    for epoch in range(1, args.epochs+1):
        t0=time.time(); tr_loss, tr_acc = train_epoch(model, loaders['train'], device, opt, criterion)
        f1, auc, report, cm = evaluate(model, loaders['val'], device)
        dt=time.time()-t0
        print(f"Epoch {epoch}/{args.epochs} loss {tr_loss:.4f} acc {tr_acc:.3f} f1 {f1:.3f} auc {auc:.3f} time {dt:.1f}s")
        if f1>best_f1:
            best_f1=f1
            torch.save({'model': model.state_dict(), 'num_classes': n_classes}, best_path)
            print(f"Saved best model to {best_path}")
    ckpt=torch.load(best_path, map_location=device); model.load_state_dict(ckpt['model'])
    f1, auc, report, cm = evaluate(model, loaders['test'], device)
    print('Test report:\n'+report)
    print('Confusion matrix:\n'+str(cm))
    print(f'Test F1 {f1:.3f} AUC {auc:.3f}')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch', type=int, default=12)
    p.add_argument('--imgsz', type=int, default=256)
    p.add_argument('--lr', type=float, default=2e-4)
    # Default workers=0 on Windows to avoid multiprocessing spawn/join hangs; raise if you have a stable setup
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--out', type=str, default='runs')
    p.add_argument('--balanced_sampler', action='store_true', help='Use a WeightedRandomSampler on the training set to balance classes')
    p.add_argument('--use_class_weights', action='store_true', help='Use class weights in CrossEntropyLoss computed from train set')
    args=p.parse_args(); main(args)
