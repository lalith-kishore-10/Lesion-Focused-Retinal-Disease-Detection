"""
CNN-based pipeline for lesion-focused DR detection
- Preprocessing: CLAHE, optional ESRGAN (placeholder hook)
- Segmentation: Optional U-Net mask application (hook)
- Classifier: Compact CNN suited for fundus images
- Metrics: Lesion-level sensitivity (approx via CAM), AUC, F1

Usage (Windows PowerShell):
python cnn_model.py --data_root "path/to/dataset" --epochs 10 --batch 16 --imgsz 224

Notes:
- Replace ESRGAN and U-Net hooks with actual implementations if available.
- Dataset expected folder structure:
  data_root/
    train/ class0/..., class1/...
    val/   class0/..., class1/...
    test/  class0/..., class1/...
"""
import os
import argparse
import time
from pathlib import Path
from typing import Tuple, List


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
import torchvision.models as tvm
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, precision_recall_fscore_support

# --------- Preprocessing utilities ---------

def apply_clahe_rgb(img_bgr: np.ndarray, clip=2.0, tile=(8,8)) -> np.ndarray:
    """Apply CLAHE to each channel in LAB space to boost contrast without blowing colors."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    out = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return out


def center_crop_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    m = min(h, w)
    y0 = (h - m) // 2
    x0 = (w - m) // 2
    return img[y0:y0+m, x0:x0+m]


def preprocess_image_bgr(img_bgr: np.ndarray, out_size: int = 224) -> np.ndarray:
    img = apply_clahe_rgb(img_bgr)
    img = center_crop_square(img)
    img = cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return img


class CLAHEDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None, size=224):
        super().__init__(root)
        self.base_transform = transform
        self.size = size

    def __getitem__(self, index):
        path, target = self.samples[index]
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        img_bgr = preprocess_image_bgr(img_bgr, out_size=self.size)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = transforms.ToTensor()(img_rgb)
        if self.base_transform is not None:
            img = self.base_transform(img)
        return img, target


# --------- Model ---------

class SmallFundusCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        feat = self.features(x)
        logits = self.classifier(feat)
        return logits


# --------- Training & evaluation ---------

def get_loaders(data_root: Path, img_size: int, batch: int, num_workers: int=2, use_balanced_sampler: bool=False):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        norm,
    ])
    base = transforms.Compose([norm])

    ds_train = CLAHEDataset(data_root/"train", transform=aug, size=img_size)
    ds_val = CLAHEDataset(data_root/"val", transform=base, size=img_size)
    ds_test = CLAHEDataset(data_root/"test", transform=base, size=img_size)

    # Only pin memory when CUDA is available. On CPU-only machines PyTorch warns that
    # pin_memory won't be used — avoid that by gating the flag.
    pin_mem = torch.cuda.is_available()

    # Optionally create a WeightedRandomSampler for the training set to mitigate
    # class imbalance. When used, set shuffle=False because the sampler controls sampling.
    if use_balanced_sampler:
        # ds_train.targets is available because CLAHEDataset inherits ImageFolder
        train_targets = ds_train.targets
        class_counts = np.bincount(train_targets)
        # prevent division by zero
        class_counts[class_counts == 0] = 1
        sample_weights = np.array([1.0 / class_counts[t] for t in train_targets], dtype=np.float32)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(ds_train, batch_size=batch, sampler=sampler, num_workers=num_workers, pin_memory=pin_mem)
    else:
        train_loader = DataLoader(ds_train, batch_size=batch, shuffle=True, num_workers=num_workers, pin_memory=pin_mem)

    loaders = {
        'train': train_loader,
        'val': DataLoader(ds_val, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=pin_mem),
        'test': DataLoader(ds_test, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=pin_mem),
    }
    num_classes = len(ds_train.classes)
    return loaders, num_classes


def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum/total, correct/total


def evaluate(model, loader, device):
    model.eval()
    all_logits, all_y = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            all_logits.append(logits.cpu())
            all_y.append(y)
    logits = torch.cat(all_logits)
    y = torch.cat(all_y)
    probs = F.softmax(logits, dim=1).numpy()
    y_np = y.numpy()
    preds = probs.argmax(axis=1)
    # Use precision_recall_fscore_support with zero_division=0 to avoid
    # UndefinedMetricWarning when some classes have no predicted samples.
    p, r, f, sup = precision_recall_fscore_support(y_np, preds, average='macro', zero_division=0)
    f1 = f
    try:
        if probs.shape[1] == 2:
            auc = roc_auc_score(y_np, probs[:,1])
        else:
            auc = roc_auc_score(y_np, probs, multi_class='ovr')
    except Exception:
        auc = float('nan')
    # Ensure classification_report also uses zero_division to avoid warnings
    report = classification_report(y_np, preds, output_dict=False, zero_division=0)
    cm = confusion_matrix(y_np, preds)

    # If a class has zero predicted samples, print a friendly hint (helps debugging label issues)
    pred_counts = np.bincount(preds, minlength=probs.shape[1])
    if (pred_counts == 0).any():
        print("Warning: at least one class has zero predicted samples. Pred counts:", pred_counts)
        print("True label distribution:", np.bincount(y_np, minlength=probs.shape[1]))

    return f1, auc, report, cm


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = Path(args.data_root)
    loaders, num_classes = get_loaders(data_root, args.imgsz, args.batch, args.workers, use_balanced_sampler=args.balanced_sampler)
    if num_classes < 2:
        raise SystemExit(
            f"Found only {num_classes} class in '{data_root}'. Classification requires >=2 classes. "
            "If your images are all in one flat folder, either: (a) create subfolders per label, or (b) use the splitter with a CSV mapping, or (c) choose a different model for one-class anomaly detection."
        )

    model = SmallFundusCNN(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Optionally compute class weights from the training set and pass to CrossEntropyLoss
    if args.use_class_weights:
        train_targets = loaders['train'].dataset.targets
        class_counts = np.bincount(train_targets)
        class_counts[class_counts == 0] = 1
        # weight = total_samples / (num_classes * count)
        total = float(train_targets.__len__())
        weights = total / (len(class_counts) * class_counts.astype(np.float32))
        weights = torch.tensor(weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print('Using class weights for loss:', weights.cpu().numpy())
    else:
        criterion = nn.CrossEntropyLoss()
    best_f1, best_path = 0.0, Path(args.out) / 'cnn_best.pt'
    Path(args.out).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, loaders['train'], device, optimizer, criterion)
        f1, auc, report, cm = evaluate(model, loaders['val'], device)
        dt = time.time()-t0
        print(f"Epoch {epoch}/{args.epochs} | loss {tr_loss:.4f} | acc {tr_acc:.3f} | f1 {f1:.3f} | auc {auc:.3f} | {dt:.1f}s")
        if f1 > best_f1:
            best_f1 = f1
            torch.save({'model': model.state_dict(), 'num_classes': num_classes}, best_path)
            print(f"Saved new best to {best_path}")

    # Final test using best model
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    f1, auc, report, cm = evaluate(model, loaders['test'], device)
    print("Test classification report:\n" + report)
    print("Confusion matrix:\n" + str(cm))
    print(f"Test F1: {f1:.3f} | AUC: {auc:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data', help="Path to dataset root with train/val/test subfolders (default: 'data')")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=224)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--out', type=str, default='runs')
    parser.add_argument('--balanced_sampler', action='store_true', help='Use a WeightedRandomSampler on the training set to balance classes')
    parser.add_argument('--use_class_weights', action='store_true', help='Use class weights in CrossEntropyLoss computed from train set')
    args = parser.parse_args()
    # Helpful validation for common path issues
    dr = Path(args.data_root)
    if not dr.exists():
        raise SystemExit(f"--data_root '{dr}' not found. If you used tools/split_dataset.py with --dst data from project root, run from project root and set --data_root data, or provide an absolute path.")
    for split in ['train','val','test']:
        if not (dr / split).exists():
            raise SystemExit(f"Missing subfolder: {dr / split}. Expected ImageFolder structure: {dr}/train|val|test/<class>/images")
    main(args)
